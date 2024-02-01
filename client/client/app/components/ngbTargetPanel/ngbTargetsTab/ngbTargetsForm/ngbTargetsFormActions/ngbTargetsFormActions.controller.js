export default class ngbTargetsFormActionsController {

    allColumns = [];

    get columnsList() {
        return this.allColumns.filter(c => (
            !this.ngbTargetGenesTableService.defaultColumns.includes(c.name)
        ));
    }

    static get UID() {
        return 'ngbTargetsFormActionsController';
    }

    constructor($scope, $timeout, dispatcher, ngbTargetGenesTableService, ngbTargetsFormService) {
        Object.assign(this, {$scope, $timeout, dispatcher, ngbTargetGenesTableService, ngbTargetsFormService});

        dispatcher.on('target:form:reset:columns', this.resetColumnsSelection.bind(this));
        dispatcher.on('target:form:table:columns', this.setColumnsList.bind(this));
        $scope.$on('$destroy', () => {
            dispatcher.removeListener('target:form:reset:columns', this.resetColumnsSelection.bind(this));
            dispatcher.removeListener('target:form:table:columns', this.setColumnsList.bind(this));
        });
    }

    setColumnsList() {
        this.allColumns = this.ngbTargetGenesTableService.allColumns.map(c => {
            const column = { name: c };
            if (this.ngbTargetGenesTableService.currentColumns.includes(c)) {
                column.selection = true;
            }
            return column;
        });
    }

    get displayFilters() {
        return this.ngbTargetGenesTableService.displayFilters;
    }
    set displayFilters(value) {
        this.ngbTargetGenesTableService.displayFilters = value;
    }

    async onChangeShowFilters() {
        this.dispatcher.emit('target:form:filters:display:changed');
        await this.ngbTargetGenesTableService.onChangeShowFilters()
            .then(result => {
                if (result) {
                    this.dispatcher.emit('target:form:filters:list');
                }
            });
    }

    resetColumnsSelection() {
        for (let i = 0; i < this.allColumns.length; i++) {
            this.allColumns[i].selection = false;
        }
    }

    onClickRestore() {
        if (this.ngbTargetsFormService.needSaveGeneChanges()) {
            this.dispatcher.emit('target:form:restore:view');
        } else {
            localStorage.setItem('targetGenesColumnsOrder', null);
            localStorage.setItem('targetGenesColumns', null);
            this.ngbTargetGenesTableService.restoreView();
            this.resetColumnsSelection();
            this.dispatcher.emit('target:form:filters:display:changed');
            this.$timeout(() => this.$scope.$apply());
        }
    }

    onChangeColumn() {
        const currentSelectedColumns = this.ngbTargetGenesTableService.additionalColumns;
        const newSelectedColumns = this.columnsList
            .filter(column => column.selection === true)
            .map(m => m.name);
        const [added] = newSelectedColumns.filter(i => currentSelectedColumns.indexOf(i) === -1);
        const [removed] = currentSelectedColumns.filter(c => newSelectedColumns.indexOf(c) === -1);
        if (added) {
            const columns = [...currentSelectedColumns];
            columns.push(added)
            this.ngbTargetGenesTableService.setAdditionalColumns(columns);
            this.ngbTargetGenesTableService.setFilterList(added);
        }
        if (removed) {
            const columns = [...currentSelectedColumns];
            const index = columns.indexOf(removed);
            if (index >= 0) {
                columns.splice(index, 1);
                this.ngbTargetGenesTableService.setAdditionalColumns(columns);
            }
        }
        this.dispatcher.emit('target:genes:columns:changed');
    }
}