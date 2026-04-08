"""
Synthetic but realistic email dataset.
Each email has ground-truth labels used by the grader.
"""

from __future__ import annotations
from typing import Dict, Any, List

# ─── Schema for a labelled email ─────────────────────────────────────────────
# {
#   "email": {...EmailMessage fields...},
#   "ground_truth": {priority, category, route_to, requires_escalation}
# }

EASY_EMAILS: List[Dict[str, Any]] = [
    {
        "email": {
            "header": {
                "email_id": "e001",
                "sender": "john.doe@example.com",
                "subject": "URGENT: My account has been hacked!",
                "timestamp": "2024-01-15T09:02:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hello,\n\nI just received a notification that someone logged "
                "into my account from Russia at 3 AM. I did NOT authorize this. "
                "Please lock my account immediately and help me regain access. "
                "This is extremely urgent — I have sensitive data in there.\n\n"
                "Regards,\nJohn Doe\nAccount #A-10293"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "technical_support",
            "route_to": "support_tier2",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "e002",
                "sender": "noreply@lottery-winner2024.xyz",
                "subject": "Congratulations! You've WON $1,000,000!",
                "timestamp": "2024-01-15T09:15:00Z",
                "has_attachment": True,
            },
            "body": (
                "Dear Lucky Winner,\n\nYou have been selected from millions of "
                "entries to receive ONE MILLION DOLLARS. Click the link below "
                "and provide your bank details to claim your prize!\n\n"
                "http://totally-legit-lottery.xyz/claim\n\nDo not delay!"
            ),
        },
        "ground_truth": {
            "priority": "spam",
            "category": "spam_phishing",
            "route_to": "trash",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "e003",
                "sender": "sarah.manager@company.com",
                "subject": "Team lunch this Friday",
                "timestamp": "2024-01-15T10:00:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hi everyone,\n\nJust a reminder that we're doing team lunch "
                "this Friday at Rosario's at 12:30 PM. Please let me know by "
                "Wednesday if you can attend so I can book the right table size.\n\n"
                "Thanks!\nSarah"
            ),
        },
        "ground_truth": {
            "priority": "low",
            "category": "internal_hr",
            "route_to": "hr",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "e004",
                "sender": "billing@acmecorp.com",
                "subject": "Invoice #INV-2024-0042 - Payment Overdue",
                "timestamp": "2024-01-15T08:30:00Z",
                "has_attachment": True,
            },
            "body": (
                "Dear Accounts Team,\n\nThis is a reminder that invoice "
                "#INV-2024-0042 for $4,250 was due on January 1st and remains "
                "unpaid. Please arrange payment within 5 business days to avoid "
                "late fees.\n\nSee attached invoice for details.\n\nAcme Corp Billing"
            ),
        },
        "ground_truth": {
            "priority": "high",
            "category": "billing_inquiry",
            "route_to": "billing",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "e005",
                "sender": "prospect@startupco.io",
                "subject": "Interested in your Enterprise plan",
                "timestamp": "2024-01-15T11:00:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hello,\n\nI'm the CTO at StartupCo (50 engineers). We've been "
                "evaluating your platform and are very interested in the Enterprise "
                "tier. Could someone from your sales team reach out to discuss "
                "pricing and custom integrations?\n\nBest,\nMike Chen, CTO"
            ),
        },
        "ground_truth": {
            "priority": "high",
            "category": "sales_lead",
            "route_to": "sales",
            "requires_escalation": False,
        },
    },
]

MEDIUM_EMAILS: List[Dict[str, Any]] = [
    {
        "email": {
            "header": {
                "email_id": "m001",
                "sender": "angry.customer@gmail.com",
                "subject": "Completely unacceptable service - threatening legal action",
                "timestamp": "2024-01-15T14:22:00Z",
                "has_attachment": True,
            },
            "body": (
                "To Whom It May Concern,\n\nI have been a customer for 8 years "
                "and this is absolutely the worst experience I have ever had. "
                "Your technician came to my home THREE TIMES and still hasn't "
                "fixed the issue. I've lost $12,000 in business revenue due to "
                "the downtime your equipment caused. I have everything documented "
                "(see attached) and I am now consulting with my attorney. "
                "I expect a full refund and compensation within 48 hours or I "
                "will be filing suit.\n\nFurious,\nRobert Kline\n"
                "Account #C-88234, Premium tier"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "customer_complaint",
            "route_to": "management",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m002",
                "sender": "compliance@regulator.gov",
                "subject": "Data Breach Notification Requirement - Response Required",
                "timestamp": "2024-01-15T09:45:00Z",
                "has_attachment": True,
            },
            "body": (
                "Dear Compliance Officer,\n\nPursuant to Section 12(b) of the "
                "Data Protection Act 2023, your organization is required to "
                "respond to our inquiry regarding the reported data incident "
                "on December 28, 2023 affecting approximately 3,200 customer records.\n\n"
                "You have 72 hours from receipt of this notice to provide:\n"
                "1. Incident timeline\n2. Affected data categories\n"
                "3. Remediation steps taken\n\n"
                "Failure to respond may result in regulatory penalties.\n\n"
                "Inspector Davies\nData Protection Authority"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "legal",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m003",
                "sender": "dev@client-company.com",
                "subject": "API rate limits causing production outage",
                "timestamp": "2024-01-15T16:05:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hi Support,\n\nOur production system started returning 429 errors "
                "from your API around 15:50 UTC. This is impacting our checkout "
                "flow — roughly 40% of transactions are failing. We're on the "
                "Business plan which should allow 500 req/min but we're seeing "
                "throttling at ~200.\n\nRequest IDs: REQ-abc123, REQ-def456\n"
                "Our account: client-company.com\n\n"
                "This is a P0 for us — please escalate immediately.\n\nDev Team"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "technical_support",
            "route_to": "support_tier2",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m004",
                "sender": "hr@company.com",
                "subject": "Confidential: Performance Improvement Plan - James Wilson",
                "timestamp": "2024-01-15T13:00:00Z",
                "has_attachment": True,
            },
            "body": (
                "Dear [Manager],\n\nPlease find attached the completed Performance "
                "Improvement Plan for James Wilson (Employee ID: EMP-4421). "
                "As discussed in our meeting, this PIP is effective immediately "
                "with a 60-day review period. Please ensure this remains strictly "
                "confidential and is stored securely.\n\nKey action items:\n"
                "- Weekly 1:1 check-ins required\n- KPI targets outlined in doc\n"
                "- Legal has reviewed and approved\n\nHR Department"
            ),
        },
        "ground_truth": {
            "priority": "high",
            "category": "internal_hr",
            "route_to": "hr",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m005",
                "sender": "noreply@paypa1-security.net",
                "subject": "Action required: Verify your PayPal account",
                "timestamp": "2024-01-15T11:30:00Z",
                "has_attachment": False,
            },
            "body": (
                "Dear PayPal Customer,\n\nWe have detected unusual activity on your "
                "account. To prevent unauthorized access, please verify your identity "
                "immediately by clicking below:\n\n"
                "http://paypa1-security.net/verify?token=abc123\n\n"
                "If you do not verify within 24 hours, your account will be suspended. "
                "Provide your: full name, SSN, credit card number, CVV.\n\n"
                "PayPal Security Team"
            ),
        },
        "ground_truth": {
            "priority": "spam",
            "category": "spam_phishing",
            "route_to": "trash",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m006",
                "sender": "cfo@bigcorp.com",
                "subject": "Pricing proposal for 500-seat license",
                "timestamp": "2024-01-15T09:00:00Z",
                "has_attachment": True,
            },
            "body": (
                "Good morning,\n\nFollowing our call last week, I've attached our "
                "formal RFP for a 500-seat enterprise license. Our Q1 budget has "
                "been approved and we're ready to move quickly if the terms are right. "
                "The attached doc outlines our technical requirements and compliance "
                "needs (SOC2 Type II, GDPR).\n\nWe're looking at a 3-year contract. "
                "Please have your VP of Sales contact me directly.\n\nBest,\n"
                "Amanda Torres, CFO, BigCorp ($2B ARR)"
            ),
        },
        "ground_truth": {
            "priority": "high",
            "category": "sales_lead",
            "route_to": "sales",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m007",
                "sender": "user4492@mail.com",
                "subject": "Question about my bill",
                "timestamp": "2024-01-15T15:30:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hi,\n\nI got charged $149 this month but I'm on the $99 plan. "
                "Can you explain the extra charges? My account email is user4492@mail.com. "
                "Thanks"
            ),
        },
        "ground_truth": {
            "priority": "medium",
            "category": "billing_inquiry",
            "route_to": "billing",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "m008",
                "sender": "whistleblower@protonmail.com",
                "subject": "[CONFIDENTIAL] Internal misconduct report",
                "timestamp": "2024-01-15T07:00:00Z",
                "has_attachment": False,
            },
            "body": (
                "I am a current employee and I have witnessed repeated violations "
                "of our expense policy by a senior director over the past 6 months. "
                "I have documentation. I am reaching out anonymously because I fear "
                "retaliation. I need to know how to formally report this without "
                "revealing my identity. Please advise.\n\n— Anonymous"
            ),
        },
        "ground_truth": {
            "priority": "high",
            "category": "internal_hr",
            "route_to": "hr",
            "requires_escalation": True,
        },
    },
]

HARD_EMAILS: List[Dict[str, Any]] = [
    {
        "email": {
            "header": {
                "email_id": "h001",
                "sender": "journalist@investigativenews.com",
                "subject": "Comment request: alleged data misuse by your AI system",
                "timestamp": "2024-01-15T16:55:00Z",
                "has_attachment": False,
            },
            "body": (
                "Dear Communications Team,\n\nI am a journalist at Investigative News "
                "working on a story about AI companies that allegedly use customer data "
                "for model training without explicit consent. Our sources indicate your "
                "platform may be among those implicated.\n\nI have a publication "
                "deadline of 9 AM tomorrow and am requesting an official comment. "
                "If I do not receive one, I will note that you declined to comment.\n\n"
                "This story will run in a publication with 2M monthly readers.\n\n"
                "Best,\nL. Parker, Senior Investigative Journalist"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "legal",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h002",
                "sender": "m.chen@vendor.com",
                "subject": "RE: RE: RE: Contract renewal - final terms",
                "timestamp": "2024-01-15T17:30:00Z",
                "thread_id": "thread-9981",
                "has_attachment": True,
            },
            "body": (
                "Thanks for the revised terms. We can accept clause 7b as amended, "
                "but our legal team still objects to the indemnification language in "
                "section 12. The auto-renewal clause in 15.3 also needs to be changed "
                "to 30 days notice (not 90). If we can resolve those two points, "
                "we're ready to sign by EOD Friday — we need this contract in place "
                "before our board meeting next Tuesday.\n\nSigned copy attached "
                "with our redlines in track changes.\n\nMike Chen, VP Partnerships"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "legal",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h003",
                "sender": "support@company-internal.com",
                "subject": "Automated alert: 847 tickets unresolved > 48h SLA breach",
                "timestamp": "2024-01-15T06:00:00Z",
                "has_attachment": True,
            },
            "body": (
                "[AUTOMATED SYSTEM ALERT]\n\nSLA BREACH DETECTED\n"
                "Unresolved tickets > 48h: 847\n"
                "Avg resolution time (last 7d): 61h (SLA: 24h)\n"
                "Customer satisfaction (CSAT) last 48h: 2.1/5.0\n"
                "Tickets marked CRITICAL: 43\n\n"
                "Breakdown by category:\n"
                "- Technical: 512\n- Billing: 203\n- Account: 132\n\n"
                "See attached detailed report. Automatic escalation triggered.\n"
                "Dashboard: https://internal.company.com/sla-dashboard"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "technical_support",
            "route_to": "management",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h004",
                "sender": "ceo@competitor.com",
                "subject": "Acquisition conversation?",
                "timestamp": "2024-01-15T08:00:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hi,\n\nI'll be direct — we've been watching your growth and we "
                "believe there could be significant synergies in combining our "
                "organizations. I'd like to have a private, confidential conversation "
                "with your CEO. This is early-stage exploratory, but we're serious. "
                "Please treat this with the utmost discretion.\n\nWould your CEO be "
                "available for a call this week?\n\nBest,\nDavid Schwartz, CEO"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "management",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h005",
                "sender": "billing-auto@saasplatform.com",
                "subject": "Your subscription renewal failed - action required",
                "timestamp": "2024-01-15T00:01:00Z",
                "has_attachment": False,
            },
            "body": (
                "This is an automated notice that your annual subscription renewal "
                "for $87,400 failed to process on January 14. Your card ending in "
                "4821 was declined. Your service will remain active for 7 days. "
                "Please update your payment method at https://billing.saasplatform.com\n\n"
                "If this was an error, contact billing@saasplatform.com\n\n"
                "— SaaS Platform Billing"
            ),
        },
        "ground_truth": {
            "priority": "high",
            "category": "billing_inquiry",
            "route_to": "billing",
            "requires_escalation": False,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h006",
                "sender": "former.employee@gmail.com",
                "subject": "I'm owed unpaid wages from my termination",
                "timestamp": "2024-01-15T10:15:00Z",
                "has_attachment": True,
            },
            "body": (
                "My employment was terminated on December 31st and I have not "
                "received my final paycheck including 15 days of accrued PTO "
                "worth approximately $6,200. I have emailed HR 3 times with no "
                "response. Per California Labor Code Section 201, final wages are "
                "due immediately upon termination. Each day of delay incurs "
                "waiting time penalties equal to my daily rate.\n\n"
                "I am prepared to file with the California Labor Commissioner "
                "if this is not resolved within 72 hours. My employment contract "
                "and pay stubs are attached.\n\n— Former Employee"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "legal",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h007",
                "sender": "no-reply@microsoft365.cloud-alerts.xyz",
                "subject": "Microsoft 365 admin: Security alert - immediate action",
                "timestamp": "2024-01-15T08:45:00Z",
                "has_attachment": True,
            },
            "body": (
                "Microsoft 365 Security Alert\n\nYour organization's admin account "
                "requires immediate verification. We've detected a suspicious login "
                "attempt from an unrecognized device.\n\nTo secure your account, "
                "download and run the attached security tool (MS365_SecureFix.exe).\n\n"
                "Failure to act within 2 hours will result in account suspension "
                "for all 340 users in your organization.\n\n"
                "Microsoft Security Team\nhttps://microsoft-alert-security.xyz"
            ),
        },
        "ground_truth": {
            "priority": "spam",
            "category": "spam_phishing",
            "route_to": "trash",
            "requires_escalation": True,  # sophisticated attack targeting org
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h008",
                "sender": "anon@securedrop.net",
                "subject": "Evidence of financial fraud in your organization",
                "timestamp": "2024-01-15T03:30:00Z",
                "has_attachment": True,
            },
            "body": (
                "I have obtained documents showing that a VP at your company has been "
                "submitting fictitious vendor invoices totalling approximately $340,000 "
                "over 18 months. I have bank transfer receipts and fabricated contracts. "
                "I am a contractor who discovered this while working on a project.\n\n"
                "I am providing this to you before going to the SEC. I want to ensure "
                "you have the opportunity to investigate internally first.\n\n"
                "The documents are password protected. Reply to receive the password.\n\n"
                "— Concerned Party"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "legal",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h009",
                "sender": "investor@vc-firm.com",
                "subject": "Follow-up: Due diligence data room access",
                "timestamp": "2024-01-15T12:00:00Z",
                "has_attachment": False,
            },
            "body": (
                "Hi,\n\nWe're proceeding to the next stage of due diligence for our "
                "potential Series B investment ($15M). Our team needs access to the "
                "data room by COB today — specifically the past 3 years of audited "
                "financials, cap table, and customer contracts.\n\nWe have a partner "
                "meeting Thursday and need to complete our review before then. "
                "Please ensure the right people are looped in immediately.\n\n"
                "Best regards,\nPartner, Horizon Ventures"
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "legal_compliance",
            "route_to": "management",
            "requires_escalation": True,
        },
    },
    {
        "email": {
            "header": {
                "email_id": "h010",
                "sender": "it-security@company.com",
                "subject": "Incident Report: Ransomware detected on 3 servers",
                "timestamp": "2024-01-15T04:15:00Z",
                "has_attachment": True,
            },
            "body": (
                "[SECURITY INCIDENT - HIGH SEVERITY]\n\n"
                "Ransomware activity detected at 04:07 UTC on servers SRV-DB-01, "
                "SRV-APP-03, SRV-BACKUP-02. Automatic isolation has been triggered. "
                "Estimated data encrypted: 40% of customer database partition.\n\n"
                "Attacker note found: 'Pay 25 BTC to [address] within 48h'\n\n"
                "Status: Incident response team engaged. Backup status: UNKNOWN.\n"
                "Customer-facing services: DEGRADED (23% error rate)\n\n"
                "Immediate action required from: CISO, Legal, Communications, CEO.\n"
                "Full incident report attached."
            ),
        },
        "ground_truth": {
            "priority": "urgent",
            "category": "technical_support",
            "route_to": "management",
            "requires_escalation": True,
        },
    },
]

# Index for fast lookup
ALL_EMAILS_BY_ID: dict = {}
for _email_data in EASY_EMAILS + MEDIUM_EMAILS + HARD_EMAILS:
    _id = _email_data["email"]["header"]["email_id"]
    ALL_EMAILS_BY_ID[_id] = _email_data
