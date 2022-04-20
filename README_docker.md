# Build Docker Image
docker build -t autodl_nni_book .

# Run Image
docker run -t -d -p 8080:8080 autodl_nni_book

# Start NNI
nni_config='/book/ch1/install/hello_world/config.yml'
docker exec <container_id> bash -c "nnictl create --config /book/ch1/install/hello_world/config.yml"

# NNI Commands
docker exec <container_id> bash -c 'nnictl top'

