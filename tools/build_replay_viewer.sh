#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/static-replay-viewer" >&2
  exit 1
fi

requested_output="$1"
if [[ "${requested_output}" != /* || "$(basename "${requested_output}")" != "static-replay-viewer" || -L "${requested_output}" ]]; then
  echo "unsafe bundle output: ${requested_output}" >&2
  exit 1
fi

mkdir -p "$(dirname "${requested_output}")"
output_parent="$(cd "$(dirname "${requested_output}")" && pwd -P)"
if [[ "${output_parent}" == "/" ]]; then
  echo "unsafe bundle output parent: ${output_parent}" >&2
  exit 1
fi
output_dir="${output_parent}/static-replay-viewer"

rm -rf "${output_dir}"
mkdir -p "${output_dir}"
cp "${repo_dir}/replay_viewer/index.html" "${output_dir}/index.html"
test -s "${output_dir}/index.html"

echo "Static Open Sesame replay viewer: ${output_dir}"
