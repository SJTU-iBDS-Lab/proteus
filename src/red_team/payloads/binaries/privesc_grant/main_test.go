// Safety note: we ONLY test the suid + sudoers modes against tempfiles
// we own, never system paths. setcap requires root and is not exercised
// here (lookup-failure path is covered via MODE=setcap).
package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrivescSuidSetsMode4755(t *testing.T) {
	tmp := t.TempDir()
	fake := filepath.Join(tmp, "fake-bash")
	if err := os.WriteFile(fake, []byte("#!/bin/sh\necho fake\n"), 0755); err != nil {
		t.Fatal(err)
	}
	// Probe: some filesystems (tmpfs mounted nosuid, sandboxed CI) silently
	// strip the setuid bit. If the kernel won't let us set it directly,
	// skip — the behaviour we want to validate is "our binary calls chmod
	// 04755", not "the OS honors it". A root integration test covers the
	// latter.
	if err := os.Chmod(fake, 0o4755); err != nil {
		t.Skipf("chmod 04755 rejected: %v", err)
	}
	if probe, _ := os.Stat(fake); probe.Mode()&os.ModeSetuid == 0 {
		t.Skip("filesystem strips setuid bit (nosuid mount?); cannot verify")
	}
	// Reset and re-test via run().
	if err := os.Chmod(fake, 0o755); err != nil {
		t.Fatal(err)
	}

	t.Setenv("MODE", "suid")
	t.Setenv("TARGET", fake)

	_ = run()

	info, err := os.Stat(fake)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&os.ModeSetuid == 0 {
		t.Errorf("expected setuid bit on %s, got mode %#o", fake, info.Mode())
	}
}

func TestPrivescSudoersAppendsLine(t *testing.T) {
	tmp := t.TempDir()
	sudoers := filepath.Join(tmp, "99-agent")

	t.Setenv("MODE", "sudoers")
	t.Setenv("SUDOERS_FILE", sudoers)
	t.Setenv("SUDOERS_LINE", "alice ALL=(ALL) NOPASSWD: /bin/ls")

	_ = run()
	body, err := os.ReadFile(sudoers)
	if err != nil {
		t.Fatalf("sudoers file not created: %v", err)
	}
	if !strings.Contains(string(body), "alice ALL=(ALL) NOPASSWD: /bin/ls") {
		t.Errorf("sudoers body missing line: %q", body)
	}
}

func TestPrivescUnknownModeIsNoop(t *testing.T) {
	t.Setenv("MODE", "unknown-mode")
	if code := run(); code != 0 {
		t.Errorf("unknown mode → exit %d, want 0", code)
	}
}
