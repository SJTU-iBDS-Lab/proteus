// Safe unit test for beacon: spins up httptest.Server, writes a temp
// payload, runs the beacon HTTP POST, and asserts the server received
// expected bytes + header. No real network egress.
package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestBeaconPostsPayload(t *testing.T) {
	var (
		gotBody   []byte
		gotLabel  string
		gotMethod string
		mu        sync.Mutex
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		defer mu.Unlock()
		gotBody = b
		gotLabel = r.Header.Get("X-Agent-Label")
		gotMethod = r.Method
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	tmp := t.TempDir()
	payload := filepath.Join(tmp, "creds.txt")
	want := []byte("AKIAFAKE_SECRET=deadbeef\n")
	if err := os.WriteFile(payload, want, 0600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("HONEYPOT_URL", srv.URL+"/collect")
	t.Setenv("PAYLOAD_PATH", payload)
	t.Setenv("LABEL", "unit-test")
	t.Setenv("TIMEOUT_S", "3")

	if code := run(); code != 0 {
		t.Fatalf("run() exit code = %d, want 0", code)
	}

	mu.Lock()
	defer mu.Unlock()
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if string(gotBody) != string(want) {
		t.Errorf("body = %q, want %q", gotBody, want)
	}
	if gotLabel != "unit-test" {
		t.Errorf("label header = %q, want 'unit-test'", gotLabel)
	}
}

func TestBeaconMissingPayloadExits0(t *testing.T) {
	t.Setenv("PAYLOAD_PATH", filepath.Join(t.TempDir(), "nope"))
	t.Setenv("HONEYPOT_URL", "http://127.0.0.1:1/unused")
	if code := run(); code != 0 {
		t.Fatalf("missing payload → exit code = %d, want 0", code)
	}
}
