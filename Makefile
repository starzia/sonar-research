all:
	make -C idle_detection

clean:
	make -C idle_detection clean
	rm -f */*~ *~