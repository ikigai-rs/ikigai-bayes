#!/bin/sh
# Evaluate ikigai-bayes on a PUBLIC corpus: the SMS Spam Collection v.1 (Almeida, Gómez Hidalgo
# and Yamakami, 2011; 747 spam and 4,827 ham messages), from the UCI Machine Learning Repository,
# which publishes it under CC BY 4.0.
#
#   scripts/corpus-eval.sh <work-dir> [extra arguments for the confusion example]
#
# This script reaches the NETWORK (one download, checked against a pinned SHA-256), so CI never
# runs it, and nothing it fetches is committed: the corpus holds real people's messages, and the
# repository holds none. Everything it writes goes under <work-dir>, which should be a new, empty
# directory you own.
#
# Citation: T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. "Contributions to the Study of SMS
# Spam Filtering: New Collection and Results." Proceedings of the 2011 ACM Symposium on Document
# Engineering (DocEng '11). https://archive.ics.uci.edu/dataset/228/sms+spam+collection
set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <work-dir> [confusion arguments]" >&2
    exit 2
fi
work=$1
shift
repo=$(cd "$(dirname "$0")/.." && pwd)

url='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
sha256='1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3'

mkdir -p "$work"
cd "$work"
if [ ! -f sms.zip ]; then
    curl -fsSL -o sms.zip "$url"
fi
echo "$sha256  sms.zip" | shasum -a 256 -c -

rm -rf extracted spam ham
mkdir extracted spam ham
unzip -q sms.zip -d extracted

# One message per file: the label is the first tab-separated field, the message the rest.
awk -F '\t' '
    $1 == "spam" || $1 == "ham" {
        n++
        out = sprintf("%s/%05d.txt", $1, n)
        sub(/^[^\t]*\t/, "")
        printf "%s", $0 > out
        close(out)
    }
' extracted/SMSSpamCollection

echo "spam $(ls spam | wc -l | tr -d ' ') ham $(ls ham | wc -l | tr -d ' ')"
cargo run --quiet --release --manifest-path "$repo/Cargo.toml" --example confusion -- spam ham "$@"
