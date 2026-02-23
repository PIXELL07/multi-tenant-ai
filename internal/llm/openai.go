package llm

import (
	"net/http"
)

const openAIChatURL = "https://api.openai.com/v1/chat/completions"

type OpenAIClient struct {
	apiKey string
	model  string
	client *http.Client
}
