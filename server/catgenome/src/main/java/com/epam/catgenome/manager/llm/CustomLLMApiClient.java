/*
 * MIT License
 *
 * Copyright (c) 2016-2023 EPAM Systems
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

package com.epam.catgenome.manager.llm;

import com.epam.catgenome.client.cloud.pipeline.CloudPipelineApi;
import com.epam.catgenome.client.cloud.pipeline.CloudPipelineApiBuilder;
import com.epam.catgenome.entity.llm.CustomLLMMessage;
import com.epam.catgenome.entity.llm.LLMMessage;
import com.epam.catgenome.util.QueryUtils;
import org.springframework.util.Assert;

import java.util.List;

public class CustomLLMApiClient implements CustomLLMApi {

    private CloudPipelineApi api;

    public CustomLLMApiClient(final String url, final String token) {
        this.api = new CloudPipelineApiBuilder(0, 0, url, null, token).buildClient();
    }

    @Override
    public String getSummary(final CustomLLMMessage message, final double temperature) {
        return QueryUtils.execute(api.chatMessage(message)).getResponse();
    }

    @Override
    public String getChatResponse(final List<LLMMessage> messages, final double temperature) {
        Assert.notNull(api, "Custom LLM api is not configured.");
        return QueryUtils.execute(api.chatMessage(
                        new CustomLLMMessage(messages.get(messages.size() - 1).getContent())))
                .getResponse();
    }
}
