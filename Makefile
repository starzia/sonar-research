all:
	make -C idle_detection

clean:
	make -C idle_detection clean
	rm -f */*~ *~

tarball:
	rm -Rf sonarPM_v1.0
	mkdir sonarPM_v1.0
	mkdir sonarPM_v1.0/recorder
	mkdir sonarPM_v1.0/idle_detection
	cp README Makefile sonarPM_v1.0/
	cp sonarPM sonarPM_v1.0/
	cp recorder/recorder.py sonarPM_v1.0/recorder/
#	cp idle_detection/idle_detection.i386 sonarPM_v1.0/idle_detection/idle_detection
	cp idle_detection/idle_detection.c sonarPM_v1.0/idle_detection/
	cp idle_detection/Makefile sonarPM_v1.0/idle_detection/
	tar czvf sonarPM_v1.0.tar.gz sonarPM_v1.0/