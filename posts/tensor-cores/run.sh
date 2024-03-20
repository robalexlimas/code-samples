dir=`pwd`
echo "Working on $dir"
make clean
make
./simpleTensorCoreGEMM 16 > log.log
