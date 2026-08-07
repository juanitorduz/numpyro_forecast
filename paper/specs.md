# Paper About Out of stock modeling with probabilistic forecasting

I want to write a paper about out of stock modeling with probabilistic forecasting. The idea is to present the problem and have three techniques do to this depending on the data. These three use cases are:

- docs/examples/availability_tsb.ipynb
- docs/examples/censored_demand.ipynb
- docs/examples/fresh_retail_stockout_2.ipynb

each of these examples show different techniques on how to do this depending ont the stockoutt data available.

Highlight that all these examples are available to reproduce in NumPyro Forecast.

For availability_tsb, explain the croston and tsb methods first and then explain the new method. If available compare the result plots.

For the fresh_retail_stockout_2 remark this was ran on GPU in Modal and took under 10 minutes. Hence, ensuring these techniques scales. 

## Format

- Author: Juan Orduz (juanitorduz@gmail.com)
- LaTeX with math article style.
- Extract the relevant images from the jupyter notebooks and include them in the paper.
- use bibtex for the references (in a separate file for references)

## Audience

Technical forecasting practitioneers. Hence, be explicit about the math formulas and model specifications.

## Style

Rigurous paper showing empirical examples of the techniques. Still, we want to also focus on the practical aspects and business / applications impact.

## References

- Find relevant references on the subjects and double check they exists (no hallucination)