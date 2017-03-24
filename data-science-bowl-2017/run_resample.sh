#!/bin/bash
starting=$1
ending=$2
for (( i=$starting; i<$ending; i+=10 ))
	do
		j=$((i+10))
		echo "Starting resampling batch, starting at $i" 
		python resample.py $i $j
		echo "Finished resampling batch, starting at $i, ending $j" 
	done
