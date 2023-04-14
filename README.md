# LLM_Simulation


## Exp 1. Convergence of MLE 

```
> exp1.sh

```

## Exp 2. Language Understanding under one prompt

```
> python3 generate_data.py -e 0.0
> python3 myTrain.py -b 128
> python3 LanguageUnderstanding.py -l 6 -n 1000

> python3 generate_data.py -e 0.02
> python3 myTrain.py -b 128
> python3 LanguageUnderstanding.py -l 6 -n 1000

```

## Exp 3. In-context Learning

```
> python3 generate_data.py -e 0.0
> python3 myTrain.py -b 128
> python3 ICL.py -m 5 -n 1000

> python3 generate_data.py -e 0.02
> python3 myTrain.py -b 128
> python3 ICL.py -m 5 -n 1000

```