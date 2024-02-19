import angular from 'angular';

import ngbDatasetItemDownloadUrlController from './ngbDatasetItemDownloadUrl';
import ngbGoldenLayout from './ngbGoldenLayout';
import ngbHeatmap from './ngbHeatmap';
import ngbMainSettings from './ngbMainSettings';
import ngbMainToolbar from './ngbMainToolbar';
import ngbMarkdown from './ngbMarkdown';
import ngbPanelErrorList from './ngbPanelErrorList';
import ngbSearch from './ngbSearch';
import ngbUiParts from './ngbUiParts';
import ngbVariantDetails from './ngbVariantDetails';

import collapsiblePanel from './widgets/collapsiblePanel';
import textBox from './widgets/TextBox';
import ngbColorPicker from './widgets/ngbColorPicker';
import ngbTags from './ngbTags';
import ngbChat from './ngbChat';
import ngbLLM from './ngbLLM';
import ngbFloatingPanel from './ngbFloatingPanel';
import ngbPagination from './ngbPagination';

export default angular.module('SharedComponents', [
    ngbGoldenLayout,
    ngbMainToolbar,
    ngbPanelErrorList,
    ngbSearch,
    ngbMainSettings,
    ngbUiParts,
    ngbVariantDetails,
    collapsiblePanel,
    textBox,
    ngbDatasetItemDownloadUrlController,
    ngbMarkdown,
    ngbColorPicker,
    ngbHeatmap,
    ngbTags,
    ngbChat,
    ngbLLM,
    ngbFloatingPanel,
    ngbPagination,
]).name;
