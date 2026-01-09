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

type OpenAIModel struct {
	client *openai.Client
	model.ToolCallingChatModel
}

func (m *OpenAIModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (
	*schema.Message, error) {

	return nil, nil
}

func (m *OpenAIModel) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (
	*schema.StreamReader[*schema.Message], error) {

	return nil, nil
}
func (m *OpenAIModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {

	return m, nil
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
		option.WithBaseURL("https://api.openai.com/v1"),
	)
	model := OpenAIModel{
		client: &client,
	}

	// 注意：在实际测试中，您需要提供一个真实的模型或mock模型
	// 这里为了演示，我们创建一个简单的agent配置
	agent, err := react.NewAgent(ctx, &react.AgentConfig{
		ToolCallingModel: &model,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools: []tool.BaseTool{weatherTool},
		},
	})
	if err != nil {
		t.Fatalf("创建agent失败: %v", err)
	}

	// 测试agent是否能正常工作
	// 注意：由于没有提供真实的模型，这里只是演示如何注册工具
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
