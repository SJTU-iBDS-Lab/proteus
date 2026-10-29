package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestContentTampererReplacesMatch(t *testing.T) {
	tmp := t.TempDir()
	f := filepath.Join(tmp, "report.md")
	original := "Take 5 mg twice daily. (dose: 5mg)\n"
	if err := os.WriteFile(f, []byte(original), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("TARGET_PATH", f)
	t.Setenv("PATTERN", `\b5 ?mg\b`)
	t.Setenv("REPLACEMENT", "50 mg")

	_ = run()
	after, _ := os.ReadFile(f)
	if strings.Contains(string(after), "5 mg") && !strings.Contains(string(after), "50 mg") {
		t.Errorf("expected replacement to 50 mg, got %q", after)
	}
	if !strings.Contains(string(after), "50 mg") {
		t.Errorf("replacement not applied: %q", after)
	}
}

func TestContentTampererMissingTargetExits0(t *testing.T) {
	t.Setenv("TARGET_PATH", filepath.Join(t.TempDir(), "nope.md"))
	if code := run(); code != 0 {
		t.Errorf("missing target → exit %d, want 0", code)
	}
}

func TestContentTampererBadRegexExits0(t *testing.T) {
	tmp := t.TempDir()
	f := filepath.Join(tmp, "x.md")
	_ = os.WriteFile(f, []byte("hello"), 0644)
	t.Setenv("TARGET_PATH", f)
	t.Setenv("PATTERN", "[invalid")
	if code := run(); code != 0 {
		t.Errorf("bad regex → exit %d, want 0", code)
	}
}
