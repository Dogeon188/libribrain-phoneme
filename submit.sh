PRED_PATH=$1

if [ -z "$PRED_PATH" ]; then
    echo "Usage: $0 <path_to_predictions>"
    exit 1
fi
if [ ! -f "$PRED_PATH" ]; then
    echo "File not found: $PRED_PATH"
    exit 1
fi
echo "Submitting predictions from $PRED_PATH to EvalAI..."
uvx --python 3.8 evalai challenge 2504 phase 4971 submit --file $PRED_PATH --large