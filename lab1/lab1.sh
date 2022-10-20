# lab1.sh

DATA_DIR=data
OUTPUT_DIR=output

if [ $# -gt 0 ]
then
    NAMES=$*
else
    NAMES=(praisy lc res) 
fi

VIDEOS=(person_convergence boat_slow birds)

for name in ${NAMES[*]}
do
    source_file_path=lab1_${name}.py
    if [ ! -f ${source_file_path} ]
    then
        echo "Source file \"${source_file_path}\" does not exist."
        exit
    fi
done

if [ ! -d ${DATA_DIR} ]
then
    echo "Data directory \"${DATA_DIR}\" does not exist."
    exit
fi
for video in ${VIDEOS[*]}
do
    data_file_path=${DATA_DIR}/${video}.mov
    if [ ! -f ${data_file_path} ]
    then
        echo "Data file \"${data_file_path}\" does not exist."
        exit
    fi
done

mkdir -p ${OUTPUT_DIR}

echo "Names: ${NAMES[*]}"
echo "Videos: ${VIDEOS[*]}"

for name in ${NAMES[*]}
do
    for video in ${VIDEOS[*]}
    do
        echo "Processing \"${name}_${video}\"…"
        python lab1_${name}.py ${DATA_DIR}/${video}.mov 10

        for image in *.png
        do
            mv ${image} ${OUTPUT_DIR}/${name}_${video}_${image}
        done
    done
done
