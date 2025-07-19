#!/bin/bash

TOTAL=0
PASSED=0

# Use find and a classic while loop without subshell issue
SCRIPTS=$(find scripts/ -type f -name "*.sh" | sort)

for script in $SCRIPTS; do
    if [[ -f "$script" ]]; then
        ((TOTAL++))
        echo "🔧 Running: $script"

        if bash "$script"; then
            echo "✅ Passed: $script"
            ((PASSED++))
        else
            echo "❌ Failed: $script"
        fi

        echo
    fi
done

echo "=============================="
echo "$PASSED / $TOTAL scripts passed"
echo "=============================="

# Exit non-zero if any scripts failed
if [[ $PASSED -ne $TOTAL ]]; then
    exit 1
fi

