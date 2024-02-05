import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.serve:app", host="0.0.0.0", port=50201, log_level="trace")