<div align="center">
    <h1>
        Inferring Cultural Values from Population Reading Activity
    </h1>
    <i>
        This repository is under active development and things will break
    </i>
</div>

--------------------------------------------------------------------

Research project into inferring cultural values (European Social Survey response distributions) from population reading activity (daily 1000 most read articles in associated Wikipedia project).
As algorithmically curated content consumption proliferates, a mapping between content and values is increasingly important.
This an exploration of that.

### Replication and usage
The particular experiment on which the findings in <a href="https://bsc.syrkis.com">bsc.syrkis.com</a> are based can be replicated with `docker run syrkis/bsc train`. A model can be trained to predict any numerical question asked in rounds 7, 8 and 9 of the European Social Survey, by running `docker run syrkis/bsc train --target [ess_column]`. More intertingly, the values of a given Wikipedia activity sample (1000 articles and there assocaited view counts) can be assigned an assocaited human value matrix through `docker run syrkis/bsc infer --country [country] --date [date]`.
