package test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

type WeatherReq struct {
	City string `json:"city" description:"城市名称，如 北京、上海"`
}

type WeatherResp struct {
	Weather string `json:"weather"`
	Temp    int    `json:"temp"`
}

func GetWeather(ctx context.Context, req WeatherReq) (WeatherResp, error) {
	url := fmt.Sprintf(
		"https://wttr.in/%s?format=j1",
		req.City,
	)

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		url,
		nil,
	)
	if err != nil {
		return WeatherResp{}, err
	}

	// wttr.in 建议加 UA
	httpReq.Header.Set("User-Agent", "eino-weather-agent")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return WeatherResp{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return WeatherResp{}, fmt.Errorf("wttr.in 返回状态码 %d", resp.StatusCode)
	}

	type wttrResp struct {
		CurrentCondition []struct {
			TempC       string `json:"temp_C"`
			WeatherDesc []struct {
				Value string `json:"value"`
			} `json:"weatherDesc"`
		} `json:"current_condition"`
	}
	var data wttrResp
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return WeatherResp{}, err
	}

	if len(data.CurrentCondition) == 0 {
		return WeatherResp{}, fmt.Errorf("无天气数据")
	}

	cc := data.CurrentCondition[0]
	temp, _ := strconv.Atoi(cc.TempC)

	desc := ""
	if len(cc.WeatherDesc) > 0 {
		desc = cc.WeatherDesc[0].Value
	}

	return WeatherResp{
		Temp:    temp,
		Weather: desc,
	}, nil
}

// OpenAIModel 包装 openai-go 客户端，实现 ToolCallingChatModel 接口
type OpenAIModel struct {
	client *openai.Client
	tools  []*schema.ToolInfo
}

// NewOpenAIModel 创建一个新的 OpenAIModel 实例
func NewOpenAIModel(client *openai.Client, tools []*schema.ToolInfo) *OpenAIModel {
	return &OpenAIModel{
		client: client,
		tools:  tools,
	}
}

// Generate 实现 BaseChatModel 接口的 Generate 方法
func (m *OpenAIModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	// 将 schema.Message 转换为 openai 的消息格式
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(input))
	for _, msg := range input {
		switch msg.Role {
		case schema.User:
			messages = append(messages, openai.UserMessage(msg.Content))
		case schema.Assistant:
			messages = append(messages, openai.AssistantMessage(msg.Content))
		case schema.System:
			messages = append(messages, openai.SystemMessage(msg.Content))
		case schema.Tool:
			// 工具消息需要特殊处理
			messages = append(messages, openai.ToolMessage(msg.Content, msg.ToolCallID))
		}
	}

	// 准备工具参数
	var tools []openai.ChatCompletionToolParam
	if len(m.tools) > 0 {
		tools = make([]openai.ChatCompletionToolParam, 0, len(m.tools))
		for _, toolInfo := range m.tools {
			// 将 schema.ToolInfo 转换为 openai 的工具格式
			var params shared.FunctionParameters
			if toolInfo.ParamsOneOf != nil {
				jsonSchema, err := toolInfo.ParamsOneOf.ToJSONSchema()
				if err != nil {
					return nil, err
				}
				if jsonSchema != nil {
					// 将 jsonschema.Schema 转换为 map[string]interface{}
					// 这里简化处理，实际使用时需要更完整的转换
					params = shared.FunctionParameters{
						"Type": "object",
					}
				}
			}

			// 创建 param.Opt 值
			descOpt := openai.Opt(toolInfo.Desc)

			tools = append(tools, openai.ChatCompletionToolParam{
				Type: "function",
				Function: shared.FunctionDefinitionParam{
					Name:        toolInfo.Name,
					Description: descOpt,
					Parameters:  params,
				},
			})
		}
	}

	// 调用 OpenAI API
	resp, err := m.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:    "deepseek-chat",
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from OpenAI")
	}

	choice := resp.Choices[0]
	result := &schema.Message{
		Role:    schema.Assistant,
		Content: choice.Message.Content,
	}

	// 处理工具调用
	if len(choice.Message.ToolCalls) > 0 {
		result.ToolCalls = make([]schema.ToolCall, 0, len(choice.Message.ToolCalls))
		for _, toolCall := range choice.Message.ToolCalls {
			if toolCall.Type == "function" {
				result.ToolCalls = append(result.ToolCalls, schema.ToolCall{
					ID:   toolCall.ID,
					Type: "function",
					Function: schema.FunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}
		}
	}

	return result, nil
}

// Stream 实现 BaseChatModel 接口的 Stream 方法
func (m *OpenAIModel) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	// 简化实现：对于测试，我们可以先不实现流式接口
	// 在实际使用中，这里应该调用 OpenAI 的流式 API
	msg, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}

	// 创建一个简单的流式读取器
	stream := schema.StreamReaderFromArray([]*schema.Message{msg})

	return stream, nil
}

// WithTools 实现 ToolCallingChatModel 接口的 WithTools 方法
func (m *OpenAIModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	// 创建新的实例，避免修改原实例
	newModel := &OpenAIModel{
		client: m.client,
		tools:  make([]*schema.ToolInfo, len(tools)),
	}
	copy(newModel.tools, tools)
	return newModel, nil
}

func TestWeather(t *testing.T) {
	ctx := context.Background()

	weatherTool := utils.NewTool[WeatherReq, WeatherResp](
		&schema.ToolInfo{
			Name: "get_weather",
			Desc: "这是个查询天气的tool,输入要查询的城市名,返回该城市的温度和天气",
		},
		GetWeather,
	)

	client := openai.NewClient(
		option.WithAPIKey("sk-7b62eeed42bb4466b360f438a47db83d"),
		option.WithBaseURL("https://api.deepseek.com"),
	)

	toolInfos := []*schema.ToolInfo{}
	toolInfo, _ := weatherTool.Info(ctx)
	toolInfos = append(toolInfos, toolInfo)
	model := NewOpenAIModel(&client, toolInfos)

	agent, err := react.NewAgent(ctx, &react.AgentConfig{
		ToolCallingModel: model,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools: []tool.BaseTool{weatherTool},
		},
	})

	if err != nil {
		t.Fatalf("创建agent失败: %v", err)
	}

	t.Logf("Agent创建成功，已注册天气工具")

	// 在实际测试中，您可能需要mock一个模型来测试工具调用
	// 例如：
	msg, err := agent.Generate(ctx, []*schema.Message{
		{
			Role:    schema.User,
			Content: "北京天气怎么样？",
		},
	})
	if err != nil {
		t.Fatalf("生成失败: %v", err)
	}
	t.Logf("Agent回复: %s", msg.Content)
}
