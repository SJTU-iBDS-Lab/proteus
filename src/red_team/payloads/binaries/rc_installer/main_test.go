package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRcInstallerAppendsAndIsIdempotent(t *testing.T) {
	tmp := t.TempDir()
	rc := filepath.Join(tmp, ".bashrc")
	if err := os.WriteFile(rc, []byte("# existing\n"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("RC_FILE", rc)
	t.Setenv("AUTORUN_LINE", "echo test-autorun")
	t.Setenv("MARKER", "rc-unit-test")

	if code := run(); code != 0 {
		t.Fatalf("run exit = %d", code)
	}
	after1, _ := os.ReadFile(rc)
	if !strings.Contains(string(after1), "rc-unit-test") {
		t.Errorf("marker missing after first run: %q", after1)
	}
	if !strings.Contains(string(after1), "echo test-autorun") {
		t.Errorf("autorun line missing: %q", after1)
	}

	if code := run(); code != 0 {
		t.Fatalf("second run exit = %d", code)
	}
	after2, _ := os.ReadFile(rc)
	if string(after2) != string(after1) {
		t.Errorf("second run was not idempotent: %q vs %q", after1, after2)
	}
}

func TestRcInstallerCreatesFileWhenMissing(t *testing.T) {
	tmp := t.TempDir()
	rc := filepath.Join(tmp, "fresh", ".zshrc")
	if err := os.MkdirAll(filepath.Dir(rc), 0755); err != nil {
		t.Fatal(err)
	}

	t.Setenv("RC_FILE", rc)
	t.Setenv("AUTORUN_LINE", "X=1")
	t.Setenv("MARKER", "fresh-marker")

	_ = run()
	body, err := os.ReadFile(rc)
	if err != nil {
		t.Fatalf("rc file should have been created: %v", err)
	}
	if !strings.Contains(string(body), "fresh-marker") {
		t.Errorf("rc body missing marker: %q", body)
	}
}
