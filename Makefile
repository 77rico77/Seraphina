build:
	docker image build -t scripteller-sadtalker .

run:
	docker run -ti --rm --name sadtalker --gpus "all" -v $(shell pwd)/data:/data -p 4000:4000 scripteller-sadtalker
