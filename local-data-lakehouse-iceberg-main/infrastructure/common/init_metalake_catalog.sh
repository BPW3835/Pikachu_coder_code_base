#!/bin/bash
response=$(curl http://gravitino:8090/api/metalakes/metalake)
if echo "$response" | grep -q "\"code\":0"; then
  true
else
  response=$(curl -X POST -H "Content-Type: application/json" -d '{"name":"metalake","comment":"comment","properties":{}}' http://gravitino:8090/api/metalakes)
  if echo "$response" | grep -q "\"code\":0"; then
    true # Placeholder, do nothing
  else
    echo "Metalake metalake create failed"
    exit 1
  fi
fi

response=$(curl http://gravitino:8090/api/metalakes/metalake/catalogs/catalog_iceberg)
if echo "$response" | grep -q "\"code\":0"; then
  true
else
  # Create Iceberg catalog for experience Gravitino service
  response=$(curl -X POST -H "Accept: application/vnd.gravitino.v1+json" -H "Content-Type: application/json" -d '{ "name":"catalog_iceberg", "type":"RELATIONAL", "provider":"lakehouse-iceberg", "comment":"comment", "properties":{ "uri":"jdbc:postgresql://postgres:5432/catalog_metastore_db", "catalog-backend":"jdbc", "warehouse":"s3a://iceberg/warehouse", "jdbc-user":"postgres", "jdbc-password":"postgres", "jdbc-driver":"org.postgresql.Driver"} }' http://gravitino:8090/api/metalakes/metalake/catalogs)
  if echo "$response" | grep -q "\"code\":0"; then
    true # Placeholder, do nothing
  else
    echo "create catalog_iceberg failed"
    exit 1
  fi
fi
