FROM python:3.12-slim
WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

# only what the API needs: the shared feature code and the categorization rules
COPY features/ features/
COPY serving/ serving/
COPY dbt/credit_scoring/seeds/category_rules.csv dbt/credit_scoring/seeds/category_rules.csv

ENV PYTHONPATH=/app RAY_USAGE_STATS_ENABLED=0
EXPOSE 8000
CMD ["serve", "run", "serving/config.yaml"]
