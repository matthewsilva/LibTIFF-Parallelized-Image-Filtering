#!/bin/bash
#
#SBATCH --job-name=conv2d
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#
## number of cores on the compute node
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#
#SBATCH --time=10:00
#
#
#
#module load LibTIFF

g++ conv2d.cc -ltiff -lpthread -Wall -o prog



## the following is just an example, you can add more calls

echo "Image Filter Combination, Sequential Time, 4 Threads Time, 8 Threads Time, 16 Threads Time, 32 Threads Time, 48 Threads Time"

printf "earth-2048.tif / Gaussian-blur,"
./prog images/earth-2048.tif images/out.tif filters/gaussian-blur.txt seq 1
printf ","
./prog images/earth-2048.tif images/out.tif filters/gaussian-blur.txt par 4
printf ","
./prog images/earth-2048.tif images/out.tif filters/gaussian-blur.txt par 8
printf ","
./prog images/earth-2048.tif images/out.tif filters/gaussian-blur.txt par 16
printf ","
./prog images/earth-2048.tif images/out.tif filters/gaussian-blur.txt par 32
printf ","
./prog images/earth-2048.tif images/out.tif filters/gaussian-blur.txt par 48
printf ","
printf "\n"

printf "earth-2048.tif / Identity-7,"
./prog images/earth-2048.tif images/out.tif filters/identity-7.txt seq 1
printf ","
./prog images/earth-2048.tif images/out.tif filters/identity-7.txt par 4
printf ","
./prog images/earth-2048.tif images/out.tif filters/identity-7.txt par 8
printf ","
./prog images/earth-2048.tif images/out.tif filters/identity-7.txt par 16
printf ","
./prog images/earth-2048.tif images/out.tif filters/identity-7.txt par 32
printf ","
./prog images/earth-2048.tif images/out.tif filters/identity-7.txt par 48
printf ","
printf "\n"

printf "earth-8192.tif / Gaussian-blur,"
./prog images/earth-8192.tif images/out.tif filters/gaussian-blur.txt seq 1
printf ","
./prog images/earth-8192.tif images/out.tif filters/gaussian-blur.txt par 4
printf ","
./prog images/earth-8192.tif images/out.tif filters/gaussian-blur.txt par 8
printf ","
./prog images/earth-8192.tif images/out.tif filters/gaussian-blur.txt par 16
printf ","
./prog images/earth-8192.tif images/out.tif filters/gaussian-blur.txt par 32
printf ","
./prog images/earth-8192.tif images/out.tif filters/gaussian-blur.txt par 48
printf ","
printf "\n"

printf "earth-8192.tif / Identity-7,"
./prog images/earth-8192.tif images/out.tif filters/identity-7.txt seq 1
printf ","
./prog images/earth-8192.tif images/out.tif filters/identity-7.txt par 4
printf ","
./prog images/earth-8192.tif images/out.tif filters/identity-7.txt par 8
printf ","
./prog images/earth-8192.tif images/out.tif filters/identity-7.txt par 16
printf ","
./prog images/earth-8192.tif images/out.tif filters/identity-7.txt par 32
printf ","
./prog images/earth-8192.tif images/out.tif filters/identity-7.txt par 48
printf ","
printf "\n"
