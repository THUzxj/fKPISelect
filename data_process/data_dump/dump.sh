

NODES="192.168.121.36 192.168.121.253 192.168.121.165 192.168.121.162 192.168.121.170"

DATASET="prometheus_normal"

python data_process/data_dump/dump_prometheus.py \
    -s "2023-04-02 00:00:00" \
    -e "2023-04-16 00:00:00" \
    -o ${DATASET}_raw \
    -n $NODES


DATASET="prometheus_fi"

python data_process/data_dump/dump_prometheus.py \
    -s "2023-04-21 00:00:00" \
    -e "2023-05-01 00:00:00" \
    -o ${DATASET}_raw \
    -n $NODES


