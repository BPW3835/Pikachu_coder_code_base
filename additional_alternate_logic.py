import os
import pendulum
import asyncio
import traceback
from google.cloud import bigquery
from helpers.log import get_logger
from bq.bq_logic import BQLogic
from helpers.utility import Utility
from api.neverbounce_api_connector import NeverbounceApiConnector
from bq.models.audit_table_schema import audit_table_schema

class AdditionalAlternateLogic:
    def __init__(self,execution_id=None):
        self.logger = get_logger()
        self.bq_logic = BQLogic()
        self.utility = Utility()
        self.execution_id = execution_id
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.lookup_dataset = os.environ.get("LOOKUP_DATASET")
        self.source_lookup_table = os.environ.get("SOURCE_LOOKUP_TABLE")
        self.lookup_table = os.environ.get("LOOKUP_TABLE")
        self.udf_path = f"{self.project_id}.{os.environ.get('UDF_PATH')}"
        self.structure_merge_udf_path = os.environ.get('STRUCTURE_MERGE_UDF_PATH')
        self.logging_dataset = os.environ.get("LOGGING_DATASET")
        self.audit_table = os.environ.get("AUDIT_TABLE")
        self.max_batch_size = os.environ.get("MAX_BATCH_SIZE")
    
    def sql_nb_email_extraction_query(self):
        try:
            client = bigquery.Client(project=self.project_id)
            sql_merge_across_all_sources = """
                            BEGIN
                CREATE TEMP TABLE merge_all_sources_alternate_try AS
                WITH
                exploded_email_validity_combined AS(
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_contacts`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                UNION ALL
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_contacts_cdil`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                UNION ALL
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_sftp`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                UNION ALL
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_sftp_events`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                UNION ALL
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_cdil_contact_events`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                UNION ALL
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_hmz_leadgen.in_hmz_facebook_leads_flexible`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                UNION ALL
                SELECT
                mdlzID,
                ev.email,
                ev.validation_result,
                SAFE_CAST(ev.validation_date AS TIMESTAMP) AS validation_date
                FROM `prd-amea-cde-data-prj.in_hmz_leadgen.in_hmz_facebook_leads_flexible`
                CROSS JOIN UNNEST(`prd-amea-cde-data-prj.in_udf.parse_email_validity_js_temp_try`(TO_JSON_STRING(email_validity))) AS ev
                ),

                unique_exploded_email_validity_combined AS (
                    select * except(rn) from(
                    select mdlzID,email,validation_result,validation_date,
                    row_number() over(partition by mdlzID,email order by validation_date desc) as rn
                    from exploded_email_validity_combined
                    )
                    where rn = 1
                ),

                hmz_contacts_emails AS(
                    SELECT
                mdlzID,
                email_required
                FROM (
                SELECT
                    mdlzID,
                    `nprd-amea-cde-data-prj.in_udf.getUnvalidatedEmails`(TO_JSON_STRING(email), email_validity) AS email_required
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_contacts`
                ), UNNEST(email_required) AS email_required
                ),

                hmz_contacts_cdil_emails AS(
                    SELECT
                mdlzID,
                email_required
                FROM (
                SELECT
                    mdlzID,
                    `nprd-amea-cde-data-prj.in_udf.getUnvalidatedEmails`(TO_JSON_STRING(email), email_validity) AS email_required
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_contacts_cdil`
                ), UNNEST(email_required) AS email_required
                ),

                hmz_sftp_emails AS(
                    SELECT
                mdlzID,
                email_required
                FROM (
                SELECT
                    mdlzID,
                    `nprd-amea-cde-data-prj.in_udf.getUnvalidatedEmails`(TO_JSON_STRING(email), email_validity) AS email_required
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_sftp`
                ), UNNEST(email_required) AS email_required
                ),

                hmz_sftp_events_emails AS(
                    SELECT
                mdlzID,
                email_required
                FROM (
                SELECT
                    mdlzID,
                    `nprd-amea-cde-data-prj.in_udf.getUnvalidatedEmails`(TO_JSON_STRING(email), email_validity) AS email_required
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_sftp_events`
                ), UNNEST(email_required) AS email_required
                ),

                hmz_cdil_contacts_events_emails AS(
                    SELECT
                mdlzID,
                email_required
                FROM (
                SELECT
                    mdlzID,
                    `nprd-amea-cde-data-prj.in_udf.getUnvalidatedEmails`(TO_JSON_STRING(email), email_validity) AS email_required
                FROM `prd-amea-cde-data-prj.in_harmonized.in_hmz_cdil_contact_events`
                ), UNNEST(email_required) AS email_required
                ),               

                hmz_leadgen AS(
                    SELECT
                mdlzID,
                email_required
                FROM (
                SELECT
                    mdlzID,
                    `nprd-amea-cde-data-prj.in_udf.getUnvalidatedEmails`(TO_JSON_STRING(email), email_validity) AS email_required
                FROM `prd-amea-cde-data-prj.in_hmz_leadgen.in_hmz_facebook_leads_flexible`
                ), UNNEST(email_required) AS email_required
                ),

                hmz_all_sources_email AS(
                SELECT DISTINCT
                mdlzid,
                email_required FROM (
                SELECT * FROM hmz_contacts_emails
                UNION ALL
                SELECT * FROM hmz_contacts_cdil_emails
                UNION ALL
                SELECT * FROM hmz_sftp_emails
                UNION ALL
                SELECT * FROM hmz_sftp_events_emails
                UNION ALL
                SELECT * FROM hmz_cdil_contacts_events_emails
                UNION ALL
                SELECT * FROM hmz_leadgen
                )
                ),

                combined_validation_emails AS(
                    select mdlzid,email_required from
                    (
                    select mdlzid,
                    email_required,
                    row_number() over(partition by email_required) as rn
                    from hmz_all_sources_email
                    )
                    where rn = 1
                ),

                combined_validation_result_attached AS(
                    select a.mdlzid,a.email_required,b.validation_result,b.validation_date
                from combined_validation_emails a
                left join unique_exploded_email_validity_combined b
                on a.mdlzid = b.mdlzId
                and a.email_required = b.email
                ),


                already_validated_emails AS(
                    select mdlzid as mdlzID,email_required as email,validation_result,validation_date
                    from combined_validation_result_attached
                    where validation_result is not null
                ),

                required_email_validation AS(
                    select mdlzid as mdlzID,email_required
                    from combined_validation_result_attached
                    where validation_result is null
                )

                select * from required_email_validation;
                END"""
            query_job = client.query(sql_merge_across_all_sources)
            merge_across_all_sources_data =  query_job.result()
            self.logger.info(f"✅ Neverbounce email extraction query completed with rows {query_job.total_rows}")
            return merge_across_all_sources_data
        except Exception as e:
            traceback.print_exc()
            self.logger.error("Error in building Neverbounce email extraction SQL query: %s", str(e))
            raise
    
    def fetch_emails_as_chunked(self,sql: str,chunk_size):
        try:
            job_config = bigquery.QueryJobConfig()
            result = self.utility.execute_query(sql,job_config)
            total_rows = int(result.total_rows or 0)

            # Generator that converts each Row to dict
            def row_generator():
                for row in result:
                    yield dict(row)

            chunk_id = 0
            gen = self.utility.chunked_iterable(row_generator(), int(chunk_size))

            # Attach metadata (optional)
            try:
                setattr(gen, "total_rows", total_rows)
            except Exception:
                pass

            for batch in gen:
                chunk_id += 1
                yield (chunk_id, batch)

        except Exception as ex:
            self.logger.info("Error in fetching unvalidated emails:", ex)
            raise

    async def process_emails(self):
        validated_email_count = 0
        valid_count = 0
        invalid_count = 0
        catchall_count = 0
        unknown_count = 0
        disposable_count = 0
        start_time = pendulum.now("UTC").isoformat()
        try:
            neverbounce_api_key = self.bq_logic.initialize_config()
            neverbounce_connector = NeverbounceApiConnector(neverbounce_api_key)
            merge_across_all_sources_data = self.sql_nb_email_extraction_query()
            for chunk_id, chunk in self.fetch_emails_as_chunked(merge_across_all_sources_data, self.max_batch_size):
                self.logger.info("Processing chunk_id=%s of size=%d", chunk_id,len(chunk))
                emails = [row["email_required"] for row in chunk]
                neverbounce_results = []
                neverbounce_results = await neverbounce_connector.verify_emails_bulk(emails, chunk_id=chunk_id)
                if not isinstance(emails, list):
                        raise ValueError("Expected 'emails' list inside BigQuery call")
                validated_email_count += len(neverbounce_results)
                for result in neverbounce_results:
                    if result["result"] == 'valid':
                        valid_count += 1
                    if result["result"] == 'invalid':
                        invalid_count += 1
                    if result["result"] == 'catchall':
                        catchall_count += 1
                    if result["result"] == 'unknown':
                        unknown_count += 1
                    if result["result"] == 'disposable':
                        disposable_count += 1
                self.bq_logic.insert_in_lookup(self.project_id, self.lookup_dataset, self.lookup_table, neverbounce_results)
                self.logger.info("Neverbounce process completed for chunk_id=%s of size=%d", chunk_id,len(chunk))
            end_time = pendulum.now("UTC").isoformat()
            audit_json_list = self.bq_logic.build_audit_json_list(start_time,end_time,validated_email_count,valid_count,
                                                                  invalid_count,catchall_count,unknown_count,
                                                                  disposable_count,self.execution_id)
            self.logger.info(f"[DEBUG] AUDIT JSON BEFORE SUCCESS INSERT: {audit_json_list}")
            self.logger.info(f"[DEBUG] DATASET={self.logging_dataset}, TABLE={self.audit_table}")
            self.bq_logic.batch_insert_json_to_audit_table(audit_json_list,self.logging_dataset,self.audit_table,audit_table_schema)
                
                


        except Exception as e:
            self.logger.error(f"Neverbounce Validation failed withe error - {e}")
            end_time = pendulum.now("UTC").isoformat()
            audit_json_list = self.bq_logic.build_audit_json_list(start_time,end_time,validated_email_count,valid_count,
                                                                  invalid_count,catchall_count,unknown_count,
                                                                  disposable_count,self.execution_id)
            self.logger.error(f"[DEBUG] AUDIT JSON BEFORE FAILURE INSERT: {audit_json_list}")
            self.logger.error(f"[DEBUG] DATASET={self.logging_dataset}, TABLE={self.audit_table}")
            self.bq_logic.batch_insert_json_to_audit_table(audit_json_list,self.logging_dataset,self.audit_table,audit_table_schema)
            self.logger.error("Error in initializing Neverbounce processing: %s", str(e))
            raise
