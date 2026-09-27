"""Fetches real ClinVar missense variants (pathogenic + benign) for a small
set of well-studied genes, resolves each gene's canonical UniProt sequence,
and writes a labelled ground-truth dataset for the ESM2 scoring pipeline."""

import re
import time

import pandas as pd
import requests

GENES = ["BRCA1", "TP53", "CFTR", "MLH1", "PTEN"]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"

PROTEIN_CHANGE_RE = re.compile(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def esearch(term: str, retmax: int = 40) -> list[str]:
    resp = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params={"db": "clinvar", "term": term, "retmax": retmax, "retmode": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def esummary(ids: list[str]) -> list[dict]:
    records = []
    for i in range(0, len(ids), 150):  # POST + chunking avoids URL-length issues
        chunk = ids[i : i + 150]
        resp = requests.post(
            f"{EUTILS}/esummary.fcgi",
            data={"db": "clinvar", "id": ",".join(chunk), "retmode": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        records.extend(result[uid] for uid in result.get("uids", []))
    return records


def parse_missense(title: str) -> tuple[str, int, str] | None:
    match = PROTEIN_CHANGE_RE.search(title)
    if not match:
        return None
    wt3, pos, mt3 = match.groups()
    wt, mt = AA3_TO_1.get(wt3), AA3_TO_1.get(mt3)
    if wt is None or mt is None or wt == mt:
        return None
    return wt, int(pos), mt


def fetch_uniprot_sequence(gene: str) -> str:
    resp = requests.get(
        UNIPROT,
        params={
            "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
            "format": "json",
            "fields": "sequence,accession",
            "size": 1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    if not results:
        raise SystemExit(f"no reviewed UniProt entry found for {gene}")
    entry = results[0]
    return entry["primaryAccession"], entry["sequence"]["value"]


def fetch_gene_variants(gene: str) -> list[dict]:
    accession, sequence = fetch_uniprot_sequence(gene)
    rows = []
    for label, clinsig_term in [
        ("Pathogenic", "clinsig pathogenic"),
        ("Benign", "clinsig benign"),
    ]:
        term = f'{gene}[gene] AND "missense variant"[MCNS] AND "{clinsig_term}"[Properties]'
        ids = esearch(term)
        time.sleep(0.4)  # NCBI eutils rate limit courtesy delay
        for record in esummary(ids):
            title = record.get("title", "")
            parsed = parse_missense(title)
            if parsed is None:
                continue
            wt, pos, mt = parsed
            if pos < 1 or pos > len(sequence) or sequence[pos - 1] != wt:
                continue  # protein numbering mismatch against this isoform; skip
            rows.append(
                {
                    "gene": gene,
                    "clinvar_id": record["uid"],
                    "wt": wt,
                    "pos": pos,
                    "mt": mt,
                    "label": label,
                    "title": title,
                    "uniprot_accession": accession,
                    "sequence": sequence,
                }
            )
        time.sleep(0.4)
    return rows


def main() -> None:
    all_rows: list[dict] = []
    for gene in GENES:
        print(f"fetching {gene}...")
        all_rows.extend(fetch_gene_variants(gene))

    if not all_rows:
        raise SystemExit("no variants fetched -- check network access to NCBI/UniProt")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["gene", "pos", "wt", "mt"])
    df.to_csv("data/variants.csv", index=False)
    print(f"wrote data/variants.csv ({len(df)} variants: "
          f"{(df['label'] == 'Pathogenic').sum()} Pathogenic, "
          f"{(df['label'] == 'Benign').sum()} Benign)")


if __name__ == "__main__":
    main()
