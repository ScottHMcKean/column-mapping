# Databricks notebook source
# Setup entrypoint:
# 1. Creates metadata tables (rules, mappings, canonical columns) from repo CSVs
# 2. Generates realistic synthetic source tables across 6 systems
# 3. Validates the Vector Search endpoint and creates a Delta Sync index

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import importlib.util

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_prefix", "")
dbutils.widgets.text("rules_table", "")
dbutils.widgets.text("mappings_table", "")
dbutils.widgets.text("vs_endpoint_name", "")
dbutils.widgets.text("vs_index_name", "")
dbutils.widgets.text("embedding_model_endpoint", "")

override_catalog = dbutils.widgets.get("catalog").strip() or None
override_schema = dbutils.widgets.get("schema").strip() or None
override_table_prefix = dbutils.widgets.get("table_prefix").strip() or None
override_rules_table = dbutils.widgets.get("rules_table").strip() or None
override_mappings_table = dbutils.widgets.get("mappings_table").strip() or None
override_vs_endpoint = dbutils.widgets.get("vs_endpoint_name").strip() or None
override_vs_index = dbutils.widgets.get("vs_index_name").strip() or None
override_embedding_model = dbutils.widgets.get("embedding_model_endpoint").strip() or None

# COMMAND ----------

nb_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
if not nb_path.startswith("/Workspace/"):
    nb_path = f"/Workspace{nb_path}"
parts = nb_path.split("/")
repo_root_ws = "/".join(parts[: parts.index("notebooks")]) if "notebooks" in parts else "/".join(parts[:-1])

if importlib.util.find_spec("column_mapping") is None:
    ws_src = repo_root_ws + "/src"
    if ws_src not in sys.path:
        sys.path.insert(0, ws_src)

from column_mapping.config import compute_effective_config, load_repo_config
from column_mapping.demo_data import ensure_demo_tables
from column_mapping.vector_search import ensure_delta_sync_index, validate_endpoint, wait_for_index_ready
from column_mapping.workspace_paths import get_repo_paths

paths = get_repo_paths(dbutils)

# COMMAND ----------

repo_cfg = load_repo_config(dbutils, repo_root_ws=repo_root_ws)
cfg = compute_effective_config(
    config=repo_cfg,
    catalog=override_catalog,
    schema=override_schema,
    table_prefix=override_table_prefix,
    rules_table=override_rules_table,
    mappings_table=override_mappings_table,
    vs_endpoint_name=override_vs_endpoint,
    vs_index_name_or_full=override_vs_index,
    embedding_model_endpoint=override_embedding_model,
)

print("Creating metadata tables from CSVs...")
tables = ensure_demo_tables(
    spark=spark,
    paths=paths,
    catalog=cfg.catalog,
    schema=cfg.schema,
    table_prefix=cfg.table_prefix,
    rules_table=cfg.rules_table,
    mappings_table=cfg.mappings_table,
    canonical_columns_table=cfg.canonical_columns_table,
)

print("Metadata tables created:")
for k, v in tables.items():
    print(f"  {k}: {v}")

spark.sql(
    f"ALTER TABLE {cfg.mappings_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Enabled Change Data Feed on {cfg.mappings_table}")

# COMMAND ----------

# ==========================================================================
# Synthetic source tables
#
# Six systems of record with intentionally inconsistent column naming,
# abbreviations, casing, date formats, and data types. These are the
# tables that the column-mapping workflow discovers and standardizes.
#
# Naming patterns:
#   crm_hubspot       -- camelCase, __c custom-field suffixes, abbreviations
#   erp_netsuite      -- UPPER_SNAKE_CASE, heavy abbreviations
#   dealcloud         -- Mixed PascalCase/snake_case, domain jargon
#   stripe_payments   -- snake_case, cent-denominated amounts, unix timestamps
#   internal_dwh      -- snake_case with dim_/fact_ prefixes, _cd/_nm/_dt suffixes
#   marketo_events    -- Mixed camelCase/PascalCase, nullable everything
# ==========================================================================

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(2026)

N_ACCOUNTS = 60
N_INVOICES = 400
N_DEALS = 35
N_CONTACTS = 120
N_EVENTS = 800

account_ids = [f"ACCT-{i:04d}" for i in range(1, N_ACCOUNTS + 1)]
company_names = [
    "Meridian Capital Partners", "Apex Industrial Holdings", "Bluestone Healthcare",
    "NovaTech Solutions", "Summit Ridge Energy", "Ironclad Logistics",
    "Clearwater Financial Group", "Pinecrest Manufacturing", "Vanguard Data Systems",
    "Pacific Rim Trading Co", "Granite Peak Resources", "Sterling Biomedical",
    "Crossroads Retail Group", "Falcon Aerospace", "Bridgewater Analytics",
    "Oakmont Properties", "Redline Automotive", "Compass Navigation Inc",
    "Evergreen Sustainability", "Atlas Cloud Services", "Sentinel Security Corp",
    "Horizon Pharmaceuticals", "Keystone Infrastructure", "Ember AI Labs",
    "Northstar Ventures", "Cobalt Mining International", "Riverview Hospitality",
    "Prism Optics Ltd", "Thunderbolt Robotics", "Whiteoak Consumer Brands",
    "Cascade Water Technologies", "Ironbridge Construction", "Solaris Renewables",
    "Nexus Telecom", "Quantum Leap Biotech", "Magellan Shipping",
    "Trident Defense Systems", "Sapphire Wealth Mgmt", "Pioneer AgriTech",
    "Zenith Semiconductor", "Aurora Media Group", "Blackrock Timber",
    "Citadel Power Corp", "Diamond Edge Tools", "Eclipse Software",
    "Foxglove Therapeutics", "Goldcrest Mining", "Harbourview Real Estate",
    "Indigo Textiles", "Jade Logistics Asia", "KnightBridge Capital",
    "Luminary Education", "Mosaic Digital", "Neptune Subsea",
    "Obsidian Metals", "Paladin Cybersecurity", "Quartz Financial",
    "Ridgeline Outdoor", "Starlite Entertainment", "Tundra Exploration",
]
domains_pool = ["Technology", "Healthcare", "Energy", "Financial Services",
                "Manufacturing", "Logistics", "Real Estate", "Consumer",
                "Defense", "Agriculture", "Media", "Mining", "Telecom"]
regions_pool = ["North America", "EMEA", "APAC", "LATAM"]
countries_pool = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "SG", "CH"]
currencies = ["USD", "EUR", "GBP", "CAD", "JPY", "CHF", "AUD"]
tiers = ["Enterprise", "Mid-Market", "SMB", "Strategic"]
statuses = ["Active", "Churned", "Prospect", "Onboarding"]

def random_dates(start, end, n):
    s = pd.Timestamp(start).value // 10**9
    e = pd.Timestamp(end).value // 10**9
    return pd.to_datetime(np.random.randint(s, e, size=n), unit="s")

def random_emails(names):
    doms = ["corp.com", "global.io", "enterprise.net", "partners.co", "group.org"]
    return [
        nm.lower().replace(" ", ".").replace(",", "")[:20] + "@" + np.random.choice(doms)
        for nm in names
    ]

def random_phones(n):
    return [
        f"+1-{np.random.randint(200,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}"
        for _ in range(n)
    ]

target = f"{cfg.catalog}.{cfg.schema}"
print(f"Generating synthetic source tables in {target} ...")

# COMMAND ----------

# -- SOURCE 1: crm_hubspot (camelCase, __c suffixes, abbreviations) ---------

hubspot_companies = pd.DataFrame({
    "companyId":           account_ids,
    "companyName":         company_names,
    "acctOwner":           np.random.choice(["Sarah Chen", "Mike Rodriguez", "James Park", "Lisa Wong", "Carlos Diaz"], N_ACCOUNTS),
    "industry__c":         np.random.choice(domains_pool, N_ACCOUNTS),
    "annualRevenue__c":    np.round(np.random.lognormal(mean=15, sigma=1.5, size=N_ACCOUNTS), 2),
    "numEmployees":        np.random.choice([10, 50, 100, 250, 500, 1000, 5000, 10000], N_ACCOUNTS),
    "CustSegment":         np.random.choice(tiers, N_ACCOUNTS),
    "accountStatus":       np.random.choice(statuses, N_ACCOUNTS),
    "signupDt":            random_dates("2019-01-01", "2025-06-01", N_ACCOUNTS).strftime("%Y-%m-%d"),
    "lastContactDate":     random_dates("2024-06-01", "2026-02-01", N_ACCOUNTS).strftime("%m/%d/%Y"),
    "primaryRegion":       np.random.choice(regions_pool, N_ACCOUNTS),
    "HQ_Country":          np.random.choice(countries_pool, N_ACCOUNTS),
    "phoneNum":            random_phones(N_ACCOUNTS),
    "website_url":         [f"https://www.{n.lower().replace(' ', '')[:15]}.com" for n in company_names],
    "LeadSource":          np.random.choice(["Inbound", "Outbound", "Referral", "Event", "Partner", ""], N_ACCOUNTS),
    "NPS_Score":           np.random.choice([np.nan, *range(1, 11)], N_ACCOUNTS),
})

contact_names = [
    f"{fn} {ln}" for fn, ln in zip(
        np.random.choice(["John", "Jane", "Alex", "Maria", "Wei", "Priya", "Omar", "Sophie", "Raj", "Elena"], N_CONTACTS),
        np.random.choice(["Smith", "Johnson", "Williams", "Chen", "Patel", "Kim", "Mueller", "Garcia", "Tanaka", "Dubois"], N_CONTACTS),
    )
]

hubspot_contacts = pd.DataFrame({
    "contactId":       [f"CON-{i:05d}" for i in range(1, N_CONTACTS + 1)],
    "firstName":       [n.split()[0] for n in contact_names],
    "lastName":        [n.split()[1] for n in contact_names],
    "emailAddress":    random_emails(contact_names),
    "Phone":           random_phones(N_CONTACTS),
    "jobTitle":        np.random.choice(["CFO", "VP Finance", "Controller", "CTO", "Director IT", "Head of Data", "CEO", "COO", "Analyst"], N_CONTACTS),
    "company_id":      np.random.choice(account_ids, N_CONTACTS),
    "contact_type":    np.random.choice(["Decision Maker", "Champion", "Influencer", "End User", "Blocker"], N_CONTACTS),
    "optedOutOfEmail": np.random.choice([True, False, False, False], N_CONTACTS),
    "createDt":        random_dates("2020-01-01", "2025-12-01", N_CONTACTS).strftime("%Y-%m-%dT%H:%M:%S"),
    "LastModified":    random_dates("2025-01-01", "2026-02-01", N_CONTACTS).strftime("%Y-%m-%dT%H:%M:%S"),
})

print(f"  crm_hubspot: {len(hubspot_companies)} companies, {len(hubspot_contacts)} contacts")

# COMMAND ----------

# -- SOURCE 2: erp_netsuite (UPPER_SNAKE_CASE, heavy abbreviations) --------

inv_ids = [f"INV-{np.random.randint(100000, 999999)}" for _ in range(N_INVOICES)]

netsuite_invoices = pd.DataFrame({
    "INV_NUM":            inv_ids,
    "CUST_ACCT_ID":       np.random.choice(account_ids, N_INVOICES),
    "CUST_NM":            np.random.choice(company_names, N_INVOICES),
    "INV_DT":             random_dates("2024-01-01", "2026-01-01", N_INVOICES).strftime("%d-%b-%Y").str.upper(),
    "DUE_DT":             random_dates("2024-02-01", "2026-03-01", N_INVOICES).strftime("%d-%b-%Y").str.upper(),
    "INV_AMT":            np.round(np.random.lognormal(mean=9, sigma=1.2, size=N_INVOICES), 2),
    "TAX_AMT":            np.round(np.random.uniform(0, 5000, N_INVOICES), 2),
    "DISC_PCT":           np.random.choice([0, 0, 0, 5, 10, 15, 20], N_INVOICES),
    "NET_AMT":            0.0,
    "PMT_TERMS":          np.random.choice(["NET30", "NET45", "NET60", "NET90", "DUE_ON_RECEIPT"], N_INVOICES),
    "PMT_STATUS":         np.random.choice(["PAID", "OPEN", "OVERDUE", "PARTIAL", "VOID"], N_INVOICES),
    "GL_ACCT_CD":         np.random.choice(["4000", "4100", "4200", "4300", "4400"], N_INVOICES),
    "GL_ACCT_DESC":       np.random.choice(["Revenue - Software", "Revenue - Services", "Revenue - Support", "Revenue - Training", "Revenue - Other"], N_INVOICES),
    "DEPT_CD":            np.random.choice(["D100", "D200", "D300", "D400"], N_INVOICES),
    "DEPT_NM":            np.random.choice(["Sales", "Professional Services", "Customer Success", "Engineering"], N_INVOICES),
    "FUNC_CCY":           np.random.choice(currencies[:4], N_INVOICES),
    "RPT_CCY":            "USD",
    "EXCH_RT":            np.round(np.random.uniform(0.85, 1.35, N_INVOICES), 6),
    "PERIOD_END_DT":      np.random.choice(["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"], N_INVOICES),
    "ENTITY_CD":          np.random.choice(["US-CORP", "EU-GMBH", "UK-LTD", "CA-INC", "SG-PTE"], N_INVOICES),
    "CREATED_BY":         np.random.choice(["SYSTEM", "JSMITH", "MJONES", "ALEE"], N_INVOICES),
    "LAST_UPD_TS":        random_dates("2025-06-01", "2026-02-28", N_INVOICES).strftime("%Y%m%d%H%M%S"),
})

netsuite_invoices["NET_AMT"] = np.round(
    netsuite_invoices["INV_AMT"] * (1 - netsuite_invoices["DISC_PCT"] / 100)
    + netsuite_invoices["TAX_AMT"],
    2,
)

print(f"  erp_netsuite: {len(netsuite_invoices)} invoices")

# COMMAND ----------

# -- SOURCE 3: dealcloud (Mixed PascalCase/snake_case, PE jargon) -----------

deal_stages = [
    "Sourcing", "Initial Review", "Due Diligence", "IC Approval",
    "Term Sheet", "Signed", "Closed", "Passed", "Dead",
]
sectors = [
    "Enterprise SaaS", "FinTech", "HealthTech", "CleanTech", "Cybersecurity",
    "AI/ML", "DevOps", "EdTech", "InsurTech", "PropTech",
]

dealcloud_deals = pd.DataFrame({
    "DealID":                [f"DL-{i:04d}" for i in range(1, N_DEALS + 1)],
    "PortfolioCompany":      np.random.choice(company_names[:30], N_DEALS),
    "deal_stage":            np.random.choice(deal_stages, N_DEALS),
    "SectorFocus":           np.random.choice(sectors, N_DEALS),
    "GeographicRegion":      np.random.choice(regions_pool, N_DEALS),
    "target_country":        np.random.choice(countries_pool, N_DEALS),
    "InvestmentDate":        random_dates("2020-01-01", "2025-12-01", N_DEALS).strftime("%Y-%m-%d"),
    "entry_valuation_mm":    np.round(np.random.lognormal(mean=4, sigma=1, size=N_DEALS), 1),
    "CurrentValuation_MM":   np.round(np.random.lognormal(mean=4.5, sigma=1.2, size=N_DEALS), 1),
    "equityInvested_MM":     np.round(np.random.lognormal(mean=3, sigma=0.8, size=N_DEALS), 1),
    "RealizationDate":       [
        d if np.random.random() > 0.6 else ""
        for d in random_dates("2023-01-01", "2026-01-01", N_DEALS).strftime("%Y-%m-%d")
    ],
    "IRR_Gross":             np.round(np.random.uniform(-0.15, 0.65, N_DEALS), 4),
    "MOIC_Gross":            np.round(np.random.uniform(0.5, 4.0, N_DEALS), 2),
    "holdingPeriod_yrs":     np.round(np.random.uniform(0.5, 7, N_DEALS), 1),
    "deal_currency":         np.random.choice(currencies[:4], N_DEALS),
    "LeadPartner":           np.random.choice(["A. Morrison", "B. Nakamura", "C. Petersen", "D. Singh", "E. Hartmann"], N_DEALS),
    "board_seat":            np.random.choice(["Yes", "No", "Observer"], N_DEALS),
    "investment_thesis":     np.random.choice([
        "Market leader in niche vertical with strong recurring revenue",
        "Platform play with significant M&A upside",
        "Turnaround opportunity with new management team",
        "Category-defining technology with strong IP moat",
        "High-growth B2B SaaS with net revenue retention >130%",
    ], N_DEALS),
    "ESG_Rating":            np.random.choice(["A", "B", "C", "D", "", "N/A"], N_DEALS),
    "co_investors":          np.random.choice(["Sequoia", "Andreessen Horowitz", "KKR", "Blackstone", "None", ""], N_DEALS),
})

print(f"  dealcloud: {len(dealcloud_deals)} deals")

# COMMAND ----------

# -- SOURCE 4: stripe_payments (snake_case, cents, unix timestamps) ---------

n_payments = 600

stripe_payments = pd.DataFrame({
    "charge_id":             [f"ch_{np.random.randint(10**14, 10**15)}" for _ in range(n_payments)],
    "customer_ref":          np.random.choice(account_ids, n_payments),
    "pmt_amount_cents":      np.random.randint(1000, 5000000, n_payments),
    "payment_currency":      np.random.choice(["usd", "eur", "gbp", "cad"], n_payments),
    "payment_method_type":   np.random.choice(["card", "bank_transfer", "wire", "ach", "sepa_debit"], n_payments),
    "card_brand":            np.random.choice(["visa", "mastercard", "amex", "discover", None, None], n_payments),
    "card_last4":            [
        f"{np.random.randint(1000,9999)}" if np.random.random() > 0.3 else None
        for _ in range(n_payments)
    ],
    "created":               (pd.Timestamp("2024-01-01").value // 10**9 + np.random.randint(0, 63072000, n_payments)).tolist(),
    "captured_at":           random_dates("2024-01-01", "2026-01-01", n_payments).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "is_refunded":           np.random.choice([False, False, False, False, True], n_payments),
    "refunded_amt_cents":    0,
    "fee_amount_cents":      0,
    "net_amount_cents":      0,
    "status":                np.random.choice(["succeeded", "failed", "pending", "disputed"], n_payments, p=[0.85, 0.05, 0.05, 0.05]),
    "failure_code":          np.random.choice([None, None, None, "card_declined", "insufficient_funds", "expired_card"], n_payments),
    "description":           np.random.choice(["Invoice payment", "Subscription renewal", "One-time purchase", "Service fee", ""], n_payments),
    "metadata_order_id":     [
        f"ORD-{np.random.randint(10000,99999)}" if np.random.random() > 0.2 else None
        for _ in range(n_payments)
    ],
    "receipt_email":         [
        random_emails(["customer"])[0] if np.random.random() > 0.4 else None
        for _ in range(n_payments)
    ],
    "billing_country":       np.random.choice(countries_pool + [None], n_payments),
    "risk_score":            np.random.randint(0, 100, n_payments),
})

stripe_payments["fee_amount_cents"] = (stripe_payments["pmt_amount_cents"] * 0.029 + 30).astype(int)
stripe_payments.loc[stripe_payments["is_refunded"], "refunded_amt_cents"] = (
    stripe_payments.loc[stripe_payments["is_refunded"], "pmt_amount_cents"]
)
stripe_payments["net_amount_cents"] = (
    stripe_payments["pmt_amount_cents"]
    - stripe_payments["fee_amount_cents"]
    - stripe_payments["refunded_amt_cents"]
)

print(f"  stripe_payments: {len(stripe_payments)} charges")

# COMMAND ----------

# -- SOURCE 5: internal_dwh (snake_case, dim_/fact_ prefixes, _cd suffixes) -

dwh_customers = pd.DataFrame({
    "dim_customer_key":          np.arange(1, N_ACCOUNTS + 1),
    "src_customer_id":           account_ids,
    "src_system_cd":             np.random.choice(["CRM", "ERP", "PORTAL"], N_ACCOUNTS),
    "customer_full_name":        company_names,
    "customer_dba_name":         [
        n.split()[0] + " " + n.split()[-1] if len(n.split()) > 2 else n
        for n in company_names
    ],
    "customer_email_address":    random_emails(company_names),
    "customer_phone_number":     random_phones(N_ACCOUNTS),
    "customer_segment_cd":       np.random.choice(["ENT", "MM", "SMB", "STRAT"], N_ACCOUNTS),
    "customer_tier_cd":          np.random.choice(["PLATINUM", "GOLD", "SILVER", "BRONZE"], N_ACCOUNTS),
    "customer_status_desc":      np.random.choice(statuses, N_ACCOUNTS),
    "account_manager_name":      np.random.choice(["Sarah Chen", "Mike Rodriguez", "James Park", "Lisa Wong", "Carlos Diaz"], N_ACCOUNTS),
    "industry_vertical_nm":      np.random.choice(domains_pool, N_ACCOUNTS),
    "geo_region_nm":             np.random.choice(regions_pool, N_ACCOUNTS),
    "country_cd":                np.random.choice(countries_pool, N_ACCOUNTS),
    "primary_currency_cd":       np.random.choice(currencies[:4], N_ACCOUNTS),
    "first_order_dt":            random_dates("2019-01-01", "2024-12-01", N_ACCOUNTS).strftime("%Y-%m-%d"),
    "last_activity_dt":          random_dates("2025-01-01", "2026-02-01", N_ACCOUNTS).strftime("%Y-%m-%d"),
    "lifetime_revenue_amt":      np.round(np.random.lognormal(mean=12, sigma=2, size=N_ACCOUNTS), 2),
    "total_orders_cnt":          np.random.randint(1, 500, N_ACCOUNTS),
    "avg_deal_size_amt":         np.round(np.random.lognormal(mean=9, sigma=1, size=N_ACCOUNTS), 2),
    "churn_risk_score":          np.round(np.random.uniform(0, 1, N_ACCOUNTS), 3),
    "is_active_flag":            np.random.choice(["Y", "N"], N_ACCOUNTS, p=[0.8, 0.2]),
    "etl_load_ts":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "etl_batch_id":              np.random.randint(90000, 90100, N_ACCOUNTS),
})

print(f"  internal_dwh: {len(dwh_customers)} customers")

# COMMAND ----------

# -- SOURCE 6: marketo_events (mixed camelCase/PascalCase, nullable) --------

marketo_events = pd.DataFrame({
    "eventId":            [f"EVT-{i:06d}" for i in range(1, N_EVENTS + 1)],
    "leadId":             [f"LD-{np.random.randint(10000,99999)}" for _ in range(N_EVENTS)],
    "AccountID":          np.random.choice(account_ids + [None] * 10, N_EVENTS),
    "EventType":          np.random.choice([
        "Page View", "Form Submit", "Email Open", "Email Click",
        "Webinar Attend", "Demo Request", "Content Download",
    ], N_EVENTS),
    "eventTimestamp":     random_dates("2024-06-01", "2026-02-01", N_EVENTS).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "SessionID":          [f"sess_{np.random.randint(10**8, 10**9)}" for _ in range(N_EVENTS)],
    "pageUrl":            np.random.choice([
        "/pricing", "/demo", "/features", "/blog/ai-trends", "/case-studies",
        "/contact", "/about", "/docs/getting-started", "/webinar/register", None,
    ], N_EVENTS),
    "referrer_url":       np.random.choice([
        "https://google.com", "https://linkedin.com", "direct",
        "https://twitter.com", "https://partner-site.com", None, None, "",
    ], N_EVENTS),
    "DeviceType":         np.random.choice(["desktop", "mobile", "tablet"], N_EVENTS, p=[0.6, 0.3, 0.1]),
    "browser_name":       np.random.choice(["Chrome", "Firefox", "Safari", "Edge", None], N_EVENTS),
    "OS":                 np.random.choice(["Windows", "macOS", "iOS", "Android", "Linux", None], N_EVENTS),
    "geo_country":        np.random.choice(countries_pool + [None], N_EVENTS),
    "geo_city":           np.random.choice([
        "San Francisco", "New York", "London", "Berlin", "Tokyo", "Sydney", "Toronto", None,
    ], N_EVENTS),
    "CampaignSource":     np.random.choice(["google", "linkedin", "facebook", "email", "organic", "direct", "partner", None], N_EVENTS),
    "CampaignMedium":     np.random.choice(["cpc", "social", "email", "organic", "referral", None], N_EVENTS),
    "campaignName":       np.random.choice(["Q1_Brand_Awareness", "Product_Launch_2025", "Retargeting_Enterprise", "Webinar_Series_AI", None], N_EVENTS),
    "LeadScore":          np.random.randint(0, 100, N_EVENTS),
    "leadStage":          np.random.choice(["Prospect", "MQL", "SQL", "SAL", "Opportunity", "Customer", None], N_EVENTS),
    "UTM_Content":        np.random.choice(["hero-cta", "sidebar-banner", "footer-link", "email-header", None, None], N_EVENTS),
    "conversionValue":    np.where(
        np.random.random(N_EVENTS) > 0.7,
        np.round(np.random.uniform(50, 5000, N_EVENTS), 2),
        None,
    ),
})

print(f"  marketo_events: {len(marketo_events)} events")

# COMMAND ----------

# -- Write all synthetic source tables --------------------------------------

synthetic_sources = {
    "crm_hubspot_companies":     hubspot_companies,
    "crm_hubspot_contacts":      hubspot_contacts,
    "erp_netsuite_invoices":     netsuite_invoices,
    "dealcloud_deals":           dealcloud_deals,
    "stripe_payments":           stripe_payments,
    "dwh_customers":             dwh_customers,
    "marketo_events":            marketo_events,
}

print("\nWriting synthetic source tables:")
for tbl_name, pdf in synthetic_sources.items():
    fqn = f"{cfg.catalog}.{cfg.schema}.{tbl_name}"
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fqn)
    print(f"  {fqn}: {len(pdf)} rows, {len(pdf.columns)} columns")

total_cols = sum(len(pdf.columns) for pdf in synthetic_sources.values())
print(f"\nTotal: {len(synthetic_sources)} tables, {total_cols} columns to standardize")

# COMMAND ----------

# -- Vector Search setup ----------------------------------------------------

vsc = VectorSearchClient()

endpoint_name = validate_endpoint(vsc=vsc, endpoint_name=cfg.vs_endpoint_name)
print(f"Using Vector Search endpoint: {endpoint_name}")

# COMMAND ----------

index = ensure_delta_sync_index(
    vsc=vsc,
    endpoint_name=endpoint_name,
    index_full_name=cfg.vs_index_full_name,
    source_table_full_name=cfg.mappings_table,
    primary_key=cfg.vs_primary_key,
    embedding_source_column=cfg.vs_embedding_source_column,
    embedding_model_endpoint_name=cfg.embedding_model_endpoint,
    pipeline_type=cfg.vs_pipeline_type,
)

status = wait_for_index_ready(index=index)
print("Vector Search index ready.")
print(f"  index: {cfg.vs_index_full_name}")
print(f"  source_table: {cfg.mappings_table}")
print(f"  rows_indexed: {status.get('num_rows', 'n/a')}")
