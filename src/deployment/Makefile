run-local:
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.13 langgraph dev

run-graph:
	langgraph dev

build-graph:
	docker build --platform linux/amd64 -t mosimaiain/qijani-recommendation-engine:prod .

push-graph:
	docker push mosimaiain/qijani-recommendation-engine:prod

.PHONY: run-local build-graph run-graph push-graph