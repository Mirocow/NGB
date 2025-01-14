/*
 * MIT License
 *
 * Copyright (c) 2023 EPAM Systems
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.epam.catgenome.manager.externaldb.target.dgidb;

import com.epam.catgenome.constant.MessagesConstants;
import com.epam.catgenome.entity.externaldb.ncbi.GeneId;
import com.epam.catgenome.entity.externaldb.target.dgidb.DGIDBDrugAssociation;
import com.epam.catgenome.entity.index.FilterType;
import com.epam.catgenome.manager.externaldb.ncbi.NCBIEnsemblIdsManager;
import com.epam.catgenome.manager.externaldb.target.AbstractAssociationManager;
import com.epam.catgenome.util.FileFormat;
import lombok.SneakyThrows;
import org.apache.http.util.TextUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SortField;
import org.apache.lucene.util.BytesRef;
import org.jetbrains.annotations.Nullable;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;


@Service
public class DGIDBDrugAssociationManager extends AbstractAssociationManager<DGIDBDrugAssociation> {

    private static final int COLUMNS = 11;
    private final NCBIEnsemblIdsManager ncbiEnsemblIdsManager;

    public DGIDBDrugAssociationManager(final @Value("${targets.index.directory}") String indexDirectory,
                                       final @Value("${targets.top.hits:10000}") int targetsTopHits,
                                       final NCBIEnsemblIdsManager ncbiEnsemblIdsManager) {
        super(Paths.get(indexDirectory, "dgidb.drug.association").toString(), targetsTopHits);
        this.ncbiEnsemblIdsManager = ncbiEnsemblIdsManager;
    }

    public DGIDBDrugFieldValues getFieldValues(final List<String> geneIds)
            throws IOException, ParseException {
        final Query query = getByGeneIdsQuery(geneIds);
        final List<DGIDBDrugAssociation> result = search(query, null);
        final List<String> interactionTypes = result.stream()
                .map(DGIDBDrugAssociation::getInteractionTypes)
                .distinct()
                .sorted()
                .collect(Collectors.toList());
        final List<String> interactionClaimSources = result.stream()
                .map(DGIDBDrugAssociation::getInteractionClaimSource)
                .distinct()
                .sorted()
                .collect(Collectors.toList());
        return DGIDBDrugFieldValues.builder()
                .interactionTypes(interactionTypes)
                .interactionClaimSources(interactionClaimSources)
                .build();
    }

    @Override
    public List<DGIDBDrugAssociation> readEntries(final String path) throws IOException {
        final List<DGIDBDrugAssociation> entries = new ArrayList<>();
        String line;
        try (Reader reader = new FileReader(path); BufferedReader bufferedReader = new BufferedReader(reader)) {
            line = bufferedReader.readLine();
            String[] cells = line.split(FileFormat.TSV.getSeparator());
            Assert.isTrue(cells.length == COLUMNS, MessagesConstants.ERROR_INCORRECT_FILE_FORMAT);
            while ((line = bufferedReader.readLine()) != null) {
                cells = line.split(FileFormat.TSV.getSeparator());
                String geneId = getCellValue(cells, 2);
                String name = getCellValue(cells, 7);
                if (!TextUtils.isBlank(geneId) && !TextUtils.isBlank(name)) {
                    DGIDBDrugAssociation entry = DGIDBDrugAssociation.builder()
                            .name(name)
                            .geneId(geneId)
                            .claimName(getCellValue(cells, 5))
                            .interactionClaimSource(getCellValue(cells, 3))
                            .interactionTypes(getCellValue(cells, 4))
                            .build();
                    entries.add(entry);
                }
            }
        }
        return entries;
    }

    @Override
    public SortField getDefaultSortField() {
        return new SortField(DGIDBField.DRUG_NAME.name(), SortField.Type.STRING, false);
    }

    @SneakyThrows
    @Override
    public List<DGIDBDrugAssociation> processEntries(final List<DGIDBDrugAssociation> entries) {
        final Set<String> entrezIds = entries.stream()
                .map(DGIDBDrugAssociation::getGeneId)
                .collect(Collectors.toSet());
        final List<GeneId> geneIds = ncbiEnsemblIdsManager.searchByEntrezIds(new ArrayList<>(entrezIds));
        final Map<String, GeneId> genesMap = geneIds.stream()
                .collect(Collectors.toMap(g -> g.getEntrezId().toString(), Function.identity()));
        for (DGIDBDrugAssociation entry: entries) {
            GeneId geneId = genesMap.getOrDefault(entry.getGeneId(), null);
            if (geneId != null) {
                entry.setGeneId(geneId.getEnsemblId());
            }
        }
        return entries;
    }

    @Override
    public void addDoc(final IndexWriter writer, final DGIDBDrugAssociation entry) throws IOException {
        if (entry.getGeneId() != null && !TextUtils.isBlank(entry.getName())) {
            final Document doc = new Document();

            doc.add(new TextField(DGIDBField.GENE_ID.name(), entry.getGeneId(), Field.Store.YES));
            doc.add(new SortedDocValuesField(DGIDBField.GENE_ID.name(), new BytesRef(entry.getGeneId())));

            doc.add(new TextField(DGIDBField.DRUG_NAME.name(), entry.getName(), Field.Store.YES));
            doc.add(new SortedDocValuesField(DGIDBField.DRUG_NAME.name(), new BytesRef(entry.getName())));

            doc.add(new TextField(DGIDBField.DRUG_CLAIM_NAME.name(), entry.getClaimName(), Field.Store.YES));
            doc.add(new SortedDocValuesField(DGIDBField.DRUG_CLAIM_NAME.name(), new BytesRef(entry.getClaimName())));

            doc.add(new StringField(DGIDBField.INTERACTION_TYPES.name(), entry.getInteractionTypes(), Field.Store.YES));
            doc.add(new SortedDocValuesField(DGIDBField.INTERACTION_TYPES.name(),
                    new BytesRef(entry.getInteractionTypes())));

            doc.add(new StringField(DGIDBField.INTERACTION_CLAIM_SOURCE.name(),
                    entry.getInteractionClaimSource(), Field.Store.YES));
            doc.add(new SortedDocValuesField(DGIDBField.INTERACTION_CLAIM_SOURCE.name(),
                    new BytesRef(entry.getInteractionClaimSource())));
            writer.addDocument(doc);
        }
    }

    @Override
    public DGIDBDrugAssociation entryFromDoc(final Document doc) {
        final String name = doc.getField(DGIDBField.DRUG_NAME.name()).stringValue();
        return DGIDBDrugAssociation.builder()
                .id(name)
                .name(name)
                .url(String.format(DGIDBDrugAssociation.URL_PATTERN, name))
                .geneId(doc.getField(DGIDBField.GENE_ID.name()).stringValue())
                .interactionTypes(doc.getField(DGIDBField.INTERACTION_TYPES.name()).stringValue())
                .interactionClaimSource(doc.getField(DGIDBField.INTERACTION_CLAIM_SOURCE.name()).stringValue())
                .claimName(doc.getField(DGIDBField.DRUG_CLAIM_NAME.name()).stringValue())
                .build();
    }

    @Override
    public FilterType getFilterType(String fieldName) {
        return DGIDBField.valueOf(fieldName).getType();
    }

    @Nullable
    private static String getCellValue(final String[] cells, final int x) {
        return cells.length >= x + 1 ? cells[x].trim() : null;
    }
}
