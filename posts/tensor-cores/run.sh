dir=`pwd`
echo "Working on $dir"
make clean
make
./simpleTensorCoreGEMM 64 > log.log
