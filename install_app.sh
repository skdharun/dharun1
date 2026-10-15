
#!/bin/bash
set -eu

echo "Building the updated Student Result Predictor image..."

cd /home/ubuntu/student-result-predictor

docker build --pull -t student-app .

echo "Docker image built successfully."
