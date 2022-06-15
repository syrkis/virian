filename=bsc

pdf:
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}
	scp ${filename}.pdf syrkis@files.syrkis.com:/var/www/files.syrkis.com
	open -a Safari https://bsc.syrkis.com
	rm bsc.{aux,bbl,blg,log,out}
