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
package com.epam.catgenome.manager.externaldb.target.opentargets;

import com.epam.catgenome.constant.MessagesConstants;
import com.epam.catgenome.entity.externaldb.target.opentargets.AssociationType;
import com.epam.catgenome.entity.externaldb.target.opentargets.Disease;
import com.epam.catgenome.entity.externaldb.target.opentargets.DiseaseAssociation;
import com.epam.catgenome.manager.externaldb.target.AbstractAssociationManager;
import com.epam.catgenome.manager.index.Filter;
import com.epam.catgenome.manager.index.SortInfo;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatDocValuesField;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.BytesRef;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.epam.catgenome.util.NgbFileUtils.getDirectory;

@Service
public class DiseaseAssociationManager extends AbstractAssociationManager<DiseaseAssociation> {

    @Value("${targets.opentargets.overallScoresDir:associationByOverallDirect}")
    private String overallScoresDir;
    @Value("${targets.opentargets.scoresDir:associationByDatatypeDirect}")
    private String scoresDir;
    private final DiseaseManager diseaseManager;

    public DiseaseAssociationManager(final @Value("${targets.index.directory}") String indexDirectory,
                                     final @Value("${targets.top.hits:10000}") int targetsTopHits,
                                     final DiseaseManager diseaseManager) {
        super(Paths.get(indexDirectory, "opentargets.disease.association").toString(), targetsTopHits);
        this.diseaseManager = diseaseManager;
    }


    private List<DiseaseAssociation> searchAll(final Query query)
            throws IOException {
        final List<DiseaseAssociation> entries = new ArrayList<>();
        try (Directory index = new SimpleFSDirectory(Paths.get(indexDirectory));
             IndexReader indexReader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(indexReader);
            TopDocs topDocs = searcher.search(query, topHits);
            ScoreDoc[] scoreDocs = topDocs.scoreDocs;
            for (ScoreDoc scoreDoc : scoreDocs) {
                Document doc = searcher.doc(scoreDoc.doc);
                DiseaseAssociation entry = entryFromDoc(doc);
                entries.add(entry);
            }
        }
        return entries;
    }

    public List<DiseaseAssociation> searchAll(final List<String> geneIds)
            throws IOException, ParseException {
        final Query query = getByGeneIdsQuery(geneIds);
        return searchAll(query);
    }

    public long totalCount(final List<String> geneIds) throws ParseException, IOException {
        final List<DiseaseAssociation> result = search(geneIds, IndexFields.GENE_ID.name());
        return result.stream().map(DiseaseAssociation::getDiseaseId).distinct().count();
    }

    @Override
    public List<DiseaseAssociation> processEntries(final List<DiseaseAssociation> entries)
            throws ParseException, IOException {
        final List<DiseaseAssociation> processedEntries = convertEntries(entries);
        fillDiseaseNames(processedEntries);
        return processedEntries;
    }

    @Override
    public List<DiseaseAssociation> readEntries(final String path) throws IOException {
        final Path overallScoresPath = Paths.get(path, overallScoresDir);
        final Path scoresPath = Paths.get(path, scoresDir);
        final File directory = getDirectory(scoresPath.toString());
        final File overallDirectory = getDirectory(overallScoresPath.toString());
        final List<DiseaseAssociation> entries = new ArrayList<>();
        String line;
        final ObjectMapper objectMapper = new ObjectMapper();
        for (File f: ArrayUtils.addAll(directory.listFiles(), overallDirectory.listFiles())) {
            try (Reader reader = new FileReader(f); BufferedReader bufferedReader = new BufferedReader(reader)) {
                while ((line = bufferedReader.readLine()) != null) {
                    try {
                        JsonNode jsonNodes = objectMapper.readTree(line);
                        DiseaseAssociation entry = entryFromJson(jsonNodes);
                        entries.add(entry);
                    } catch (JsonProcessingException e) {
                        throw new IllegalStateException(MessagesConstants.ERROR_INCORRECT_JSON_FORMAT);
                    }
                }
            }
        }
        return entries;
    }

    @Override
    public String getDefaultSortField() {
        return IndexFields.DISEASE_NAME.name();
    }

    @Override
    public void addDoc(final IndexWriter writer, final DiseaseAssociation entry) throws IOException {
        final Document doc = new Document();
        addStringField(entry.getGeneId(), doc, IndexFields.GENE_ID);

        doc.add(new StringField(IndexFields.DISEASE_ID.name(), entry.getDiseaseId(), Field.Store.YES));

        addStringField(entry.getDiseaseName(), doc, IndexFields.DISEASE_NAME);

        addFloatField(entry.getOverallScore(), doc, IndexFields.OVERALL_SCORE);
        addFloatField(entry.getGeneticAssociationScore(), doc, IndexFields.GENETIC_ASSOCIATIONS_SCORE);
        addFloatField(entry.getSomaticMutationScore(), doc, IndexFields.SOMATIC_MUTATIONS_SCORE);
        addFloatField(entry.getKnownDrugScore(), doc, IndexFields.DRUGS_SCORE);
        addFloatField(entry.getAffectedPathwayScore(), doc, IndexFields.PATHWAYS_SCORE);
        addFloatField(entry.getLiteratureScore(), doc, IndexFields.TEXT_MINING_SCORE);
        addFloatField(entry.getRnaExpressionScore(), doc, IndexFields.RNA_EXPRESSION_SCORE);
        addFloatField(entry.getAnimalModelScore(), doc, IndexFields.ANIMAL_MODELS_SCORE);
        writer.addDocument(doc);
    }

    @Override
    public DiseaseAssociation entryFromDoc(final Document doc) {
        final DiseaseAssociation d = DiseaseAssociation.builder()
                .geneId(doc.getField(IndexFields.GENE_ID.name()).stringValue())
                .diseaseId(doc.getField(IndexFields.DISEASE_ID.name()).stringValue())
                .build();
        if (doc.getField(IndexFields.DISEASE_NAME.name()) != null) {
            d.setDiseaseName(doc.getField(IndexFields.DISEASE_NAME.name()).stringValue());
        }
        if (doc.getField(IndexFields.OVERALL_SCORE.name()) != null) {
            d.setOverallScore(getScore(doc, IndexFields.OVERALL_SCORE));
        }
        if (doc.getField(IndexFields.GENETIC_ASSOCIATIONS_SCORE.name()) != null) {
            d.setGeneticAssociationScore(getScore(doc, IndexFields.GENETIC_ASSOCIATIONS_SCORE));
        }
        if (doc.getField(IndexFields.SOMATIC_MUTATIONS_SCORE.name()) != null) {
            d.setSomaticMutationScore(getScore(doc, IndexFields.SOMATIC_MUTATIONS_SCORE));
        }
        if (doc.getField(IndexFields.DRUGS_SCORE.name()) != null) {
            d.setKnownDrugScore(getScore(doc, IndexFields.DRUGS_SCORE));
        }
        if (doc.getField(IndexFields.PATHWAYS_SCORE.name()) != null) {
            d.setAffectedPathwayScore(getScore(doc, IndexFields.PATHWAYS_SCORE));
        }
        if (doc.getField(IndexFields.TEXT_MINING_SCORE.name()) != null) {
            d.setLiteratureScore(getScore(doc, IndexFields.TEXT_MINING_SCORE));
        }
        if (doc.getField(IndexFields.RNA_EXPRESSION_SCORE.name()) != null) {
            d.setRnaExpressionScore(getScore(doc, IndexFields.RNA_EXPRESSION_SCORE));
        }
        if (doc.getField(IndexFields.ANIMAL_MODELS_SCORE.name()) != null) {
            d.setAnimalModelScore(getScore(doc, IndexFields.ANIMAL_MODELS_SCORE));
        }
        return d;
    }

    @Override
    public Sort getSort(final List<SortInfo> sortInfos) {
        final List<SortField> sortFields = new ArrayList<>();
        if (sortInfos == null) {
            sortFields.add(new SortField(getDefaultSortField(), SortField.Type.STRING, false));
        } else {
            for (SortInfo sortInfo: sortInfos) {
                final SortField.Type sortType = sortInfo.getOrderBy().equals(IndexFields.DISEASE_NAME.name())
                        || sortInfo.getOrderBy().equals(IndexFields.GENE_ID.name()) ?
                        SortField.Type.STRING : SortField.Type.FLOAT;
                final SortField sortField = new SortField(sortInfo.getOrderBy(),
                        sortType, sortInfo.isReverse());
                sortFields.add(sortField);
            }
        }
        return new Sort(sortFields.toArray(new SortField[sortFields.size()]));
    }

    @Override
    public void addFieldQuery(final BooleanQuery.Builder builder, final Filter filter) {
        final BooleanQuery.Builder fieldBuilder = new BooleanQuery.Builder();
        for (String term: filter.getTerms()) {
            Query query = IndexFields.GENE_ID.name().equals(filter.getField()) ||
                    IndexFields.DISEASE_NAME.name().equals(filter.getField()) ?
                    buildPrefixQuery(filter.getField(), term) : buildTermQuery(filter.getField(), term);
            fieldBuilder.add(query, BooleanClause.Occur.SHOULD);
        }
        builder.add(fieldBuilder.build(), BooleanClause.Occur.MUST);
    }

    private static DiseaseAssociation entryFromJson(final JsonNode jsonNodes) {
        return DiseaseAssociation.builder()
                .geneId(jsonNodes.at("/targetId").asText())
                .diseaseId(jsonNodes.at("/diseaseId").asText())
                .type(jsonNodes.has("datatypeId") ?
                        AssociationType.getByName(jsonNodes.at("/datatypeId").asText()) :
                        AssociationType.OVERALL)
                .score(Float.parseFloat(jsonNodes.at("/score").asText()))
                .build();
    }

    private List<DiseaseAssociation> convertEntries(final List<DiseaseAssociation> entries) {
        final List<DiseaseAssociation> results = new ArrayList<>();
        final Map<AbstractMap.SimpleEntry<String, String>, List<DiseaseAssociation>> grouped = entries.stream()
                .collect(Collectors.groupingBy(entry ->
                        new AbstractMap.SimpleEntry<>(entry.getGeneId(), entry.getDiseaseId())));
        for (Map.Entry<AbstractMap.SimpleEntry<String, String>, List<DiseaseAssociation>> group : grouped.entrySet()) {
            DiseaseAssociation result = new DiseaseAssociation();
            AbstractMap.SimpleEntry<String, String> key = group.getKey();
            result.setGeneId(key.getKey());
            result.setDiseaseId(key.getValue());
            List<DiseaseAssociation> values = group.getValue();
            for (DiseaseAssociation value: values) {
                switch (value.getType()) {
                    case OVERALL:
                        result.setOverallScore(value.getScore());
                        break;
                    case GENETIC_ASSOCIATIONS:
                        result.setGeneticAssociationScore(value.getScore());
                        break;
                    case SOMATIC_MUTATIONS:
                        result.setSomaticMutationScore(value.getScore());
                        break;
                    case DRUGS:
                        result.setKnownDrugScore(value.getScore());
                        break;
                    case PATHWAYS:
                        result.setAffectedPathwayScore(value.getScore());
                        break;
                    case TEXT_MINING:
                        result.setLiteratureScore(value.getScore());
                        break;
                    case RNA_EXPRESSION:
                        result.setRnaExpressionScore(value.getScore());
                        break;
                    case ANIMAL_MODELS:
                        result.setAnimalModelScore(value.getScore());
                        break;
                    default:
                        break;
                }
            }
            results.add(result);
        }
        return results;
    }

    private void fillDiseaseNames(final List<DiseaseAssociation> entries) throws ParseException, IOException {
        final List<String> diseaseIds = entries.stream()
                .map(DiseaseAssociation::getDiseaseId)
                .distinct()
                .collect(Collectors.toList());
        if (!CollectionUtils.isEmpty(diseaseIds)) {
            final List<Disease> diseases = diseaseManager.search(diseaseIds);
            final Map<String, Disease> diseasesMap = diseases.stream()
                    .collect(Collectors.toMap(Disease::getId, Function.identity()));
            for (DiseaseAssociation r : entries) {
                if (diseasesMap.containsKey(r.getDiseaseId())) {
                    r.setDiseaseName(diseasesMap.get(r.getDiseaseId()).getName());
                }
            }
        }
    }

    private static void addStringField(final String entry, final Document doc, final IndexFields field) {
        if (entry != null) {
            doc.add(new StringField(field.name(), entry.toLowerCase(), Field.Store.YES));
            doc.add(new SortedDocValuesField(field.name(), new BytesRef(entry)));
        }
    }

    private static void addFloatField(final Float entry, final Document doc, final IndexFields field) {
        if (entry != null) {
            doc.add(new FloatPoint(field.name(), entry));
            doc.add(new StoredField(field.name(), entry));
            doc.add(new FloatDocValuesField(field.name(), entry));
        }
    }

    private static float getScore(final Document doc, final IndexFields field) {
        return  doc.getField(field.name()).numericValue().floatValue();
    }

    @Getter
    private enum IndexFields {
        GENE_ID,
        DISEASE_ID,
        DISEASE_NAME,
        OVERALL_SCORE,
        GENETIC_ASSOCIATIONS_SCORE,
        SOMATIC_MUTATIONS_SCORE,
        DRUGS_SCORE,
        PATHWAYS_SCORE,
        TEXT_MINING_SCORE,
        RNA_EXPRESSION_SCORE,
        ANIMAL_MODELS_SCORE;
    }
}