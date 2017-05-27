#!/bin/bash

#srcPath="/home/studi2/group2/frames_211x176"
srcPath="/home/pat/Documents/UniNeuch/courses/secondSemester/AdTopML/project/age_of_romance/set/frames_38x32"

validate="validate"
train="train"
test="test"

if [ ! -d $validate ]; then mkdir $validate; fi
if [ ! -d $train ]; then mkdir $train; fi
if [ ! -d $test ]; then mkdir $test; fi

for i in {1960..2009}; do
  countMovies=1
  for movie in $(find $srcPath -type d -name "$i*"); do
    echo "ln -> $movie"
    if [ $countMovies -eq 1 ]; then ln -sT "$movie" "$validate/$i-$countMovies"; fi
    if [ $countMovies -eq 2 ]; then ln -sT "$movie" "$test/$i-$countMovies"; fi
    if [ $countMovies -ge 3 ]; then ln -sT "$movie" "$train/$i-$countMovies"; fi
    countMovies=$(($countMovies+1))
  done
done
