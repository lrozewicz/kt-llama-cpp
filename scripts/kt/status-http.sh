#!/usr/bin/env bash
# kt-llama.cpp: answers one HTTP request with 503 and the model download progress. The entrypoint runs it through
# socat (one process per connection, stdin/stdout = the connection) until llama-server takes over the port.
# The reason goes into error.message (the shape of llama-server's "Loading model") and into the status line,
# because some clients, such as requests' raise_for_status(), print only the status line.
export LC_ALL=C

len=0
while IFS= read -r -t 5 line; do  # request line and headers
    line=${line%$'\r'}
    [ -z "$line" ] && break
    [[ ${line,,} =~ ^content-length:[[:space:]]*([0-9]+) ]] && len=${BASH_REMATCH[1]}
done
# Read the body as well: closing a connection with unread data sends a reset, and the client would see only that.
[ "$len" -gt 0 ] && timeout 10 head -c "$len" >/dev/null

msg=$(tr -d '\r\n"\\' < "${KT_STATUS_FILE:-/tmp/kt-status}" 2>/dev/null)
msg=${msg:-Starting}
body="{\"error\":{\"message\":\"$msg\",\"type\":\"unavailable_error\",\"code\":503}}"
printf 'HTTP/1.1 503 %s\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: %d\r\nRetry-After: 10\r\nConnection: close\r\n\r\n%s' \
    "$msg" "${#body}" "$body"
