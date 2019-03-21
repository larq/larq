workflow "Lint" {
  on = "push"
  resolves = ["Black Code Formatter", "Pyflakes Syntax Checker"]
}

action "Black Code Formatter" {
  uses = "lgeiger/black-action@v1.0.1"
  args = ". --check"
}

action "Pyflakes Syntax Checker" {
  uses = "lgeiger/pyflakes-action@v1.0.1"
}

workflow "Unit tests" {
  on = "push"
  resolves = [
    "Python 3.6 TF 1",
    "Python 3.7 TF 1",
    "Python 3.7 TF 2",
  ]
}

action "Python 3.6 TF 1" {
  uses = "docker://python:3.6-slim"
  runs = "./scripts/test.sh"
  env = {
    TF_VERSION = "1.13.1"
  }
}

action "Python 3.7 TF 1" {
  uses = "docker://python:3.7-slim"
  runs = "./scripts/test.sh"
  env = {
    TF_VERSION = "1.13.1"
  }
}

action "Python 3.7 TF 2" {
  uses = "docker://python:3.7-slim"
  runs = "./scripts/test.sh"
  env = {
    TF_VERSION = "2.0.0-alpha0"
  }
}
