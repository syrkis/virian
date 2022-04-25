filename=paper

pdf:
	pdflatex ${filename}.tex
	bibtex ${filename}
	pdflatex ${filename}.tex
	pdflatex ${filename} 
	@if [ $(ps -a | grep Negative | wc -l)==2 ]; then\
		killall Negative && sleep 0.2;\
	fi
	open -a Negative ${filename}.pdf
	scp ${filename}.pdf syrkis@bsc.syrkis.com:/var/www/bsc.syrkis.com
