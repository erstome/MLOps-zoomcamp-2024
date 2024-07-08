create_S3_ENDPOINT_URL_env_variable:
	export S3_ENDPOINT_URL="http://localhost:4566"
	export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
	export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

create_s3_bucket:
	aws s3 --endpoint-url=$(S3_ENDPOINT_URL) mb s3://nyc-duration

list_s3_buckets:
	aws s3 --endpoint-url=$(S3_ENDPOINT_URL) ls