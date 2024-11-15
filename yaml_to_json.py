import yaml
import json

yaml_data = """
openapi: 3.0.0
info:
  title: Parliamentary Bills API
  version: '1.0'
servers:
  - url: https://your-api-url.com
paths:
  /api/bills:
    get:
      summary: Get all bills
      parameters:
        - name: house
          in: query
          schema:
            type: string
        - name: year
          in: query
          schema:
            type: string
        - name: title
          in: query
          schema:
            type: string
      responses:
        '200':
          description: List of bills
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Bill'
  /api/bills/{bill_id}:
    get:
      summary: Get bill by ID
      parameters:
        - name: bill_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Bill details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Bill'
  /api/bills/search:
    post:
      summary: Search bills
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Bill'
components:
  schemas:
    Bill:
      type: object
      properties:
        id:
          type: string
        title:
          type: string
        description:
          type: string
        date:
          type: string
        house:
          type: string
        positives:
          type: array
          items:
            $ref: '#/components/schemas/Point'
        negatives:
          type: array
          items:
            $ref: '#/components/schemas/Point'
    Point:
      type: object
      properties:
        title:
          type: string
        explanation:
          type: string
"""

# Convert YAML to JSON and save to file
json_data = json.dumps(yaml.safe_load(yaml_data), indent=4)
with open('openapi.json', 'w') as f:
    f.write(json_data)
