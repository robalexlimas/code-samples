dir=`pwd`
echo "Working on $dir"
make clean
make
./TCGemm 16 > log.log
