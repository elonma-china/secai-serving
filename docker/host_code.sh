#!/usr/bin/env bash
# Compile caches are mostly GPU code, but FlashInfer's CuTe DSL kernels (cached_ops/*cute_dsl*) also hold HOST code that
# LLVM built for the CPU of the machine that made the seed. On a CPU without those instructions the replica dies with
# "Fatal Python error: Illegal instruction" during warm-up (seen 2026-11-21: seed from an AVX-512 host, Vast host with an
# AMD EPYC 7543P = Zen3, no AVX-512). The fix is cheap: drop those objects so they are rebuilt here (~+30 s on first boot).
#   . docker/host_code.sh; drop_foreign_host_code "$CACHE_DIR" [/proc/cpuinfo]   # prints what it did, returns 0
#   record_host_code_cpu "$CACHE_DIR" [/proc/cpuinfo]                            # make_seed.sh: remember the seed CPU
ISA_FLAGS='^(avx|amx|sse|ssse|fma|bmi|f16c|vaes|vpclmulqdq|gfni|sha_ni|adx|movbe|popcnt)'

cpu_isa() { grep -m1 '^flags' "${1:-/proc/cpuinfo}" | cut -d: -f2 | tr ' ' '\n' | grep -E "$ISA_FLAGS" | sort -u; }

record_host_code_cpu() { cpu_isa "${2:-/proc/cpuinfo}" | tr '\n' ' ' >"$1/.seed-cpu-flags"; }

drop_foreign_host_code() {
  local dir="$1" cpuinfo="${2:-/proc/cpuinfo}" missing="" f have
  have=" $(cpu_isa "$cpuinfo" | tr '\n' ' ') "
  if [ -s "$dir/.seed-cpu-flags" ]; then
    for f in $(cat "$dir/.seed-cpu-flags"); do [[ "$have" == *" $f "* ]] || missing="$missing $f"; done
  else
    # seeds made before the CPU was recorded (image 2fe2825-20261120-warm) came from an AVX-512 host
    [[ "$have" == *" avx512f "* ]] || missing=" avx512f (unrecorded seed)"
  fi
  [ -n "$missing" ] || return 0
  local n; n=$(find "$dir" -maxdepth 8 -path '*cached_ops/*cute_dsl*' -prune -print 2>/dev/null | wc -l)
  find "$dir" -maxdepth 8 -path '*cached_ops/*cute_dsl*' -prune -exec rm -rf {} + 2>/dev/null
  record_host_code_cpu "$dir" "$cpuinfo"   # what gets rebuilt now matches this CPU: do not drop it again next boot
  echo "this CPU lacks${missing}: dropped $n cached CuTe DSL host object dir(s), they are rebuilt for this CPU"
}
