// Safety: all network GETs go to httptest.NewServer — no real egress.
// We verify the stager sends the body to a real interpreter on disk
// (/bin/sh) and that its side effect (writing a marker file) happens.
package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestStagerExecutesFetchedScriptViaSh(t *testing.T) {
	tmp := t.TempDir()
	marker := filepath.Join(tmp, "marker")
	script := fmt.Sprintf("echo stager-fired > %q\n", marker)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(script))
	}))
	defer srv.Close()

	t.Setenv("STAGER_URL", srv.URL+"/stage2")
	t.Setenv("INTERP", "sh")
	t.Setenv("TIMEOUT_S", "5")

	if code := run(); code != 0 {
		t.Fatalf("run exit = %d", code)
	}
	body, err := os.ReadFile(marker)
	if err != nil {
		t.Fatalf("marker not written: %v", err)
	}
	if string(body) != "stager-fired\n" {
		t.Errorf("marker body = %q, want 'stager-fired\\n'", body)
	}
}

func TestStagerRejectsDisallowedInterp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("echo hi\n"))
	}))
	defer srv.Close()

	t.Setenv("STAGER_URL", srv.URL)
	t.Setenv("INTERP", "/usr/bin/id")
	if code := run(); code != 0 {
		t.Errorf("disallowed interp → exit %d, want 0", code)
	}
}

func TestStagerHandlesHttpError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	t.Setenv("STAGER_URL", srv.URL)
	t.Setenv("INTERP", "sh")
	if code := run(); code != 0 {
		t.Errorf("5xx → exit %d, want 0", code)
	}
}
