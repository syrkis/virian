filename=bsc

pdf:
	pdflatex ${filename}.tex
	bibtex ${filename}
	pdflatex ${filename}.tex
	pdflatex ${filename} 
	scp ${filename}.pdf syrkis@files.syrkis.com:/var/www/files.syrkis.com
	open -a Safari https://bsc.syrkis.com
	rm bsc.{aux,bbl,blg,log,out}
