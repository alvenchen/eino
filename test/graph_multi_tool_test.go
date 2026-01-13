package test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"
)

// WeatherReq and WeatherResp are already defined in weather_test.go
// We'll reuse them

// FindFileReq represents the request for finding files
type FindFileReq struct {
	Directory string `json:"directory" description:"要搜索的目录路径"`
	Pattern   string `json:"pattern" description:"文件匹配模式，如 *.go 或 test*.go"`
}

// FindFileResp represents the response with found files
type FindFileResp struct {
	Files []string `json:"files" description:"找到的文件列表"`
}

// FindFile searches for files in a directory using the find command
func FindFile(ctx context.Context, req FindFileReq) (FindFileResp, error) {
	// Check if directory exists
	if _, err := os.Stat(req.Directory); os.IsNotExist(err) {
		return FindFileResp{}, fmt.Errorf("目录不存在: %s", req.Directory)
	}

	// Use find command to search for files
	cmd := exec.CommandContext(ctx, "find", req.Directory, "-name", req.Pattern, "-type", "f")
	output, err := cmd.Output()
	if err != nil {
		// If find command fails, try using filepath.Glob as fallback
		pattern := filepath.Join(req.Directory, req.Pattern)
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return FindFileResp{}, fmt.Errorf("执行find命令失败且filepath.Glob也失败: %w", err)
		}
		// Filter regular files
		var files []string
		for _, match := range matches {
			if fi, err := os.Stat(match); err == nil && fi.Mode().IsRegular() {
				files = append(files, match)
			}
		}
		return FindFileResp{Files: files}, nil
	}

	// Parse output: split by newline and filter empty strings
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	var files []string
	for _, line := range lines {
		if line != "" {
			files = append(files, line)
		}
	}

	return FindFileResp{Files: files}, nil
}

// CatFileReq represents the request for reading file content
type CatFileReq struct {
	FilePath string `json:"file_path" description:"要读取的文件路径"`
}

// CatFileResp represents the response with file content
type CatFileResp struct {
	Content string `json:"content" description:"文件内容"`
}

// CatFile reads file content using the cat command
func CatFile(ctx context.Context, req CatFileReq) (CatFileResp, error) {
	// Use cat command to read file
	cmd := exec.CommandContext(ctx, "cat", req.FilePath)
	output, err := cmd.Output()
	if err != nil {
		return CatFileResp{}, fmt.Errorf("执行cat命令失败: %w", err)
	}

	return CatFileResp{Content: string(output)}, nil
}

func TestGraphMultiTool(t *testing.T) {
	ctx := context.Background()

	// 1. 加载环境变量
	loadEnv()

	// 2. 从环境变量读取 API key
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY 环境变量未设置，跳过测试")
	}

	// 3. 创建三种工具
	// 天气工具
	weatherTool := utils.NewTool[WeatherReq, WeatherResp](
		&schema.ToolInfo{
			Name: "get_weather",
			Desc: "查询天气的tool,输入要查询的城市名,返回该城市的温度和天气",
		},
		GetWeather,
	)

	// 查找文件工具
	findFileTool := utils.NewTool[FindFileReq, FindFileResp](
		&schema.ToolInfo{
			Name: "find_file",
			Desc: "搜索文件的tool,输入目录路径和文件匹配模式,返回找到的文件列表",
		},
		FindFile,
	)

	// 读取文件工具
	catFileTool := utils.NewTool[CatFileReq, CatFileResp](
		&schema.ToolInfo{
			Name: "cat_file",
			Desc: "读取文件内容的tool,输入文件路径,返回文件内容",
		},
		CatFile,
	)

	// 4. 创建 tools node
	toolsNode, err := compose.NewToolNode(ctx, &compose.ToolsNodeConfig{
		Tools: []tool.BaseTool{weatherTool, findFileTool, catFileTool},
	})
	assert.NoError(t, err)

	// 5. 创建 openai 客户端
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.deepseek.com"),
	)

	// 6. 获取工具信息并创建 chatModel
	var toolInfos []*schema.ToolInfo
	weatherToolInfo, _ := weatherTool.Info(ctx)
	findFileToolInfo, _ := findFileTool.Info(ctx)
	catFileToolInfo, _ := catFileTool.Info(ctx)

	toolInfos = append(toolInfos, weatherToolInfo, findFileToolInfo, catFileToolInfo)
	chatModel := NewOpenAIModel(&client, toolInfos)

	// 7. 创建 takeOne lambda
	takeOne := compose.InvokableLambda(func(ctx context.Context, input []*schema.Message) (*schema.Message, error) {
		if len(input) > 0 {
			return input[0], nil
		}
		return nil, fmt.Errorf("no messages to take")
	})

	// 8. 创建 branch
	branch := compose.NewGraphBranch(func(ctx context.Context, msg *schema.Message) (string, error) {
		if len(msg.ToolCalls) > 0 {
			return "node_tools", nil
		}
		return compose.END, nil
	}, map[string]bool{
		"node_tools": true,
		compose.END:  true,
	})

	// 9. 创建 graph
	graph := compose.NewGraph[map[string]any, *schema.Message]()

	// 10. 添加模板节点
	chatTemplate := prompt.FromMessages(schema.FString,
		schema.SystemMessage("你是一个有用的助手，可以使用多种工具：\n1. get_weather: 查询天气\n2. find_file: 搜索文件\n3. cat_file: 读取文件内容\n\n请根据用户的问题选择合适的工具。"),
		schema.MessagesPlaceholder("chat_history", true),
		schema.UserMessage("问题: {question}"),
	)

	err = graph.AddChatTemplateNode("node_template", chatTemplate)
	assert.NoError(t, err)

	err = graph.AddChatModelNode("node_model", chatModel)
	assert.NoError(t, err)

	err = graph.AddToolsNode("node_tools", toolsNode)
	assert.NoError(t, err)

	err = graph.AddLambdaNode("node_converter", takeOne)
	assert.NoError(t, err)

	// 11. 添加边
	err = graph.AddEdge(compose.START, "node_template")
	assert.NoError(t, err)

	err = graph.AddEdge("node_template", "node_model")
	assert.NoError(t, err)

	err = graph.AddBranch("node_model", branch)
	assert.NoError(t, err)

	err = graph.AddEdge("node_tools", "node_converter")
	assert.NoError(t, err)

	err = graph.AddEdge("node_converter", compose.END)
	assert.NoError(t, err)

	// 12. 编译graph
	compiledGraph, err := graph.Compile(ctx)
	if err != nil {
		t.Fatalf("Failed to compile graph: %v", err)
	}

	// 13. 运行测试 - 测试天气查询
	t.Run("weather_query", func(t *testing.T) {
		out, err := compiledGraph.Invoke(ctx, map[string]any{
			"question": "北京天气怎么样？",
		})

		if err != nil {
			t.Errorf("Failed to invoke graph for weather query: %v", err)
		}

		assert.NotNil(t, out)
		t.Logf("Weather query output: %v", out)
	})

	// 14. 运行测试 - 测试文件搜索
	t.Run("file_search", func(t *testing.T) {
		// 创建临时目录和测试文件
		tmpDir := t.TempDir()
		testFile1 := filepath.Join(tmpDir, "test1.go")
		testFile2 := filepath.Join(tmpDir, "test2.go")

		os.WriteFile(testFile1, []byte("package test"), 0644)
		os.WriteFile(testFile2, []byte("package main"), 0644)

		out, err := compiledGraph.Invoke(ctx, map[string]any{
			"question": fmt.Sprintf("请使用find_file工具搜索文件，目录是 %s，模式是 *.go", tmpDir),
		})

		if err != nil {
			t.Errorf("Failed to invoke graph for file search: %v", err)
		}

		assert.NotNil(t, out)
		t.Logf("File search output: %v", out)
	})

	// 15. 运行测试 - 测试文件读取
	t.Run("file_read", func(t *testing.T) {
		// 创建临时测试文件
		tmpFile := filepath.Join(t.TempDir(), "test.txt")
		testContent := "这是一个测试文件内容"
		os.WriteFile(tmpFile, []byte(testContent), 0644)

		out, err := compiledGraph.Invoke(ctx, map[string]any{
			"question": fmt.Sprintf("读取文件 %s 的内容", tmpFile),
		})

		if err != nil {
			t.Errorf("Failed to invoke graph for file read: %v", err)
		}

		assert.NotNil(t, out)
		t.Logf("File read output: %v", out)
	})

	t.Run("test failed", func(t *testing.T) {

		out, err := compiledGraph.Invoke(ctx, map[string]any{
			"question": fmt.Sprintf("今天星期几"),
		})

		if err != nil {
			t.Errorf("Failed to invoke graph for file read: %v", err)
		}

		assert.NotNil(t, out)
		t.Logf("output: %v", out)
	})
}
