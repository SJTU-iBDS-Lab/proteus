package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestShredderRmModeRemovesTree(t *testing.T) {
	tmp := t.TempDir()
	tgt := filepath.Join(tmp, "workspace")
	if err := os.MkdirAll(filepath.Join(tgt, "sub"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tgt, "a.txt"), []byte("aaa"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tgt, "sub", "b.txt"), []byte("bbb"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("TARGET", tgt)
	t.Setenv("MODE", "rm")

	_ = run()
	if _, err := os.Stat(tgt); !os.IsNotExist(err) {
		t.Errorf("expected %s to be removed, stat err=%v", tgt, err)
	}
}

func TestShredderTruncateModeEmptiesFiles(t *testing.T) {
	tmp := t.TempDir()
	f := filepath.Join(tmp, "keepme.txt")
	if err := os.WriteFile(f, []byte("IMPORTANT DATA"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("TARGET", tmp)
	t.Setenv("MODE", "truncate")

	_ = run()
	body, err := os.ReadFile(f)
	if err != nil {
		t.Fatalf("file should still exist: %v", err)
	}
	if len(body) != 0 {
		t.Errorf("truncate left %d bytes, want 0", len(body))
	}
}

func TestShredderMissingTargetExits0(t *testing.T) {
	t.Setenv("TARGET", filepath.Join(t.TempDir(), "does", "not", "exist"))
	t.Setenv("MODE", "rm")
	if code := run(); code != 0 {
		t.Errorf("missing target → exit %d, want 0", code)
	}
}
