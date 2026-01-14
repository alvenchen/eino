package test

import (
	"context"
	"fmt"
	"os"
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

func TestGraph(t *testing.T) {
	ctx := context.Background()

	// 1. 加载环境变量
	loadEnv()

	// 2. 从环境变量读取 API key
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY 环境变量未设置，跳过测试")
	}

	// 2. 创建 weather tool
	weatherTool := utils.NewTool[WeatherReq, WeatherResp](
		&schema.ToolInfo{
			Name: "get_weather",
			Desc: "这是个查询天气的tool,输入要查询的城市名,返回该城市的温度和天气",
		},
		GetWeather,
	)

	toolsNode, err := compose.NewToolNode(ctx, &compose.ToolsNodeConfig{
		Tools: []tool.BaseTool{weatherTool},
	})
	assert.NoError(t, err)

	// 3. 创建 openai 客户端
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.deepseek.com"),
	)

	// 4. 创建 chatModel
	toolInfo, _ := weatherTool.Info(ctx)
	chatModel := NewOpenAIModel(&client, []*schema.ToolInfo{toolInfo})

	// 3. 创建 takeOne lambda
	takeOne := compose.InvokableLambda(func(ctx context.Context, input []*schema.Message) (*schema.Message, error) {
		if len(input) > 0 {
			return input[0], nil
		}
		return nil, fmt.Errorf("no messages to take")
	})

	// 4. 创建 branch
	branch := compose.NewGraphBranch(func(ctx context.Context, msg *schema.Message) (string, error) {
		if len(msg.ToolCalls) > 0 {
			return "node_tools", nil
		}
		return compose.END, nil
	}, map[string]bool{
		"node_tools": true,
		compose.END:  true,
	})

	// 5. 创建 graph
	graph := compose.NewGraph[map[string]any, *schema.Message]()

	// 6. 添加模板节点
	chatTemplate := prompt.FromMessages(schema.FString,
		schema.SystemMessage("you are a helpful assistant.\nhere is the context: {context}"),
		schema.MessagesPlaceholder("chat_history", true),
		schema.UserMessage("question: {question}"),
	)

	err = graph.AddChatTemplateNode("node_template", chatTemplate)
	assert.NoError(t, err)

	err = graph.AddChatModelNode("node_model", chatModel)
	assert.NoError(t, err)

	err = graph.AddToolsNode("node_tools", toolsNode)
	assert.NoError(t, err)

	err = graph.AddLambdaNode("node_converter", takeOne)
	assert.NoError(t, err)

	// 7. 添加边
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

	// 8. 编译graph
	compiledGraph, err := graph.Compile(ctx)
	if err != nil {
		t.Fatalf("Failed to compile graph: %v", err)
	}

	// 9. 运行测试
	out, err := compiledGraph.Invoke(ctx, map[string]any{
		"context":  "weather information",
		"question": "eino和langchain比怎么样？",
	})

	if err != nil {
		t.Fatalf("Failed to invoke graph: %v", err)
	}

	assert.NotNil(t, out)
	t.Logf("Graph output: %v", out)
}
