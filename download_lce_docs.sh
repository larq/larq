#!/usr/bin/env bash
set -e
basedir=`dirname $0`

mkdir $basedir/docs/compute-engine || true

# Fetch compute-engine docs
echo "# Larq Compute Engine" > $basedir/docs/compute-engine/index.md
curl https://raw.githubusercontent.com/larq/compute-engine/master/README.md | tail -n +4 >> $basedir/docs/compute-engine/index.md
for file in build.md build_arm.md inference.md quickstart_android.md converter.md
do
  curl https://raw.githubusercontent.com/larq/compute-engine/master/docs/$file > $basedir/docs/compute-engine/$file
done
