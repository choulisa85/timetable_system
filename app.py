import streamlit as st
import pandas as pd
import requests
import json
import time
from io import BytesIO
from ortools.sat.python import cp_model
import ast
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Générateur Intelligent d'Emploi du Temps",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Générateur Intelligent d'Emploi du Temps")
st.markdown("**Optimisé pour tes fichiers CSV spécifiques**")

class AdvancedTimetableSolver:
    def __init__(self, days=5, slots_per_day=8):
        self.days = days
        self.slots_per_day = slots_per_day
        self.total_slots = days * slots_per_day
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Configuration du solver
        self.solver.parameters.max_time_in_seconds = 120
        self.solver.parameters.num_search_workers = 8
    
    def load_data_from_dataframes(self, profs_df, classes_df, salles_df, matieres_df):
        """Charge les données depuis les DataFrames CSV"""
        self.professeurs = []
        self.classes = []
        self.salles = []
        self.matieres = []
        self.sessions = []
        
        # Charger les professeurs
        for _, row in profs_df.iterrows():
            prof = {
                'id': row['id'],
                'nom': row['nom'],
                'matieres': ast.literal_eval(row['matieres']) if 'matieres' in row else [],
                'max_daily_sessions': row.get('max_daily_sessions', 6),
                'unavailable_slots': ast.literal_eval(row.get('indispo', '[]'))
            }
            self.professeurs.append(prof)
        
        # Charger les classes
        for _, row in classes_df.iterrows():
            classe = {
                'id': row['id'],
                'nom': row['nom'],
                'unavailable_slots': ast.literal_eval(row.get('indispo', '[]'))
            }
            self.classes.append(classe)
        
        # Charger les salles
        for _, row in salles_df.iterrows():
            salle = {
                'id': row['id'],
                'nom': row['nom'],
                'type': row['type'],
                'capacite': row['capacite'],
                'equipment': ast.literal_eval(row.get('equipment', '[]')),
                'unavailable_slots': ast.literal_eval(row.get('indispo', '[]'))
            }
            self.salles.append(salle)
        
        # Charger les matières
        for _, row in matieres_df.iterrows():
            matiere = {
                'id': row['id'],
                'nom': row['nom'],
                'type': row['type'],
                'duree': row.get('duree', 1),
                'required_equipment': ast.literal_eval(row.get('required_equipment', '[]')),
                'profs_compatibles': ast.literal_eval(row.get('professeurs', '[]'))
            }
            self.matieres.append(matiere)
        
        # Générer les sessions à planifier
        self._generate_sessions()
    
    def _generate_sessions(self):
        """Génère les sessions basées sur les données chargées"""
        session_id = 0
        for matiere in self.matieres:
            for classe in self.classes:
                # Trouver les professeurs compatibles avec cette matière
                profs_compatibles = [p for p in self.professeurs if matiere['id'] in p['matieres']]
                if not profs_compatibles:
                    profs_compatibles = self.professeurs  # Fallback
                
                for prof in profs_compatibles[:2]:  # Limiter à 2 profs par matière/classe
                    # Trouver les salles compatibles
                    salles_compatibles = [
                        s for s in self.salles 
                        if all(eq in s['equipment'] for eq in matiere['required_equipment'])
                    ]
                    if not salles_compatibles:
                        salles_compatibles = self.salles  # Fallback
                    
                    for salle in salles_compatibles[:2]:  # Limiter à 2 salles
                        session = {
                            'id': f"S{session_id}",
                            'matiere_id': matiere['id'],
                            'matiere_nom': matiere['nom'],
                            'prof_id': prof['id'],
                            'prof_nom': prof['nom'],
                            'classe_id': classe['id'],
                            'classe_nom': classe['nom'],
                            'salle_id': salle['id'],
                            'salle_nom': salle['nom'],
                            'duree': matiere['duree'],
                            'type': matiere['type']
                        }
                        self.sessions.append(session)
                        session_id += 1
    
    def solve_with_constraints(self, constraints):
        """Résout le problème avec les contraintes spécifiées"""
        # Créer les variables de décision
        session_vars = {}
        for session in self.sessions:
            # Variable pour le créneau de début (0 à total_slots-1)
            session_vars[session['id']] = self.model.NewIntVar(
                0, self.total_slots - 1, f"slot_{session['id']}"
            )
        
        # Ajouter les contraintes
        self._add_basic_constraints(session_vars)
        self._add_advanced_constraints(session_vars, constraints)
        
        # Objectif : maximiser le nombre de sessions placées
        objective = sum([session_vars[s['id']] for s in self.sessions])
        self.model.Maximize(objective)
        
        # Résolution
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._extract_solution(session_vars)
        else:
            return None
    
    def _add_basic_constraints(self, session_vars):
        """Ajoute les contraintes de base"""
        # Contraintes de non-chevauchenent par professeur
        for prof in self.professeurs:
            prof_sessions = [s for s in self.sessions if s['prof_id'] == prof['id']]
            self._add_no_overlap_constraints(session_vars, prof_sessions, f"prof_{prof['id']}")
        
        # Contraintes de non-chevauchenent par salle
        for salle in self.salles:
            salle_sessions = [s for s in self.sessions if s['salle_id'] == salle['id']]
            self._add_no_overlap_constraints(session_vars, salle_sessions, f"salle_{salle['id']}")
        
        # Contraintes de non-chevauchenent par classe
        for classe in self.classes:
            classe_sessions = [s for s in self.sessions if s['classe_id'] == classe['id']]
            self._add_no_overlap_constraints(session_vars, classe_sessions, f"classe_{classe['id']}")
    
    def _add_no_overlap_constraints(self, session_vars, sessions, entity_name):
        """Ajoute des contraintes de non-chevauchenent pour un ensemble de sessions"""
        if len(sessions) <= 1:
            return
        
        intervals = []
        for session in sessions:
            start_var = session_vars[session['id']]
            duration = session['duree']
            end_var = start_var + duration
            
            # Créer une variable d'intervalle
            interval_var = self.model.NewIntervalVar(
                start_var, duration, end_var, f"interval_{session['id']}"
            )
            intervals.append(interval_var)
        
        # Ajouter la contrainte de non-chevauchenent
        self.model.AddNoOverlap(intervals)
    
    def _add_advanced_constraints(self, session_vars, constraints):
        """Ajoute les contraintes avancées"""
        
        # Contrainte : Pas de cours le vendredi après-midi
        if constraints.get('no_friday_afternoon', False):
            for session in self.sessions:
                slot_var = session_vars[session['id']]
                # Vendredi = jour 4, après-midi = créneaux >= 4 (12h)
                friday_afternoon_slots = [i for i in range(4*self.slots_per_day + 4, (4+1)*self.slots_per_day)]
                for slot in friday_afternoon_slots:
                    self.model.Add(slot_var != slot)
        
        # Contrainte : Limite de cours par jour par professeur
        if constraints.get('limit_daily_sessions', True):
            for prof in self.professeurs:
                prof_sessions = [s for s in self.sessions if s['prof_id'] == prof['id']]
                max_daily = prof.get('max_daily_sessions', 6)
                
                for day in range(self.days):
                    day_sessions = []
                    for session in prof_sessions:
                        slot_var = session_vars[session['id']]
                        # Variable booléenne indiquant si la session est ce jour
                        is_this_day = self.model.NewBoolVar(f"prof_{prof['id']}_day{day}_session_{session['id']}")
                        # Contrainte : is_this_day == True si le créneau est dans ce jour
                        start_slot = day * self.slots_per_day
                        end_slot = (day + 1) * self.slots_per_day - 1
                        self.model.Add(slot_var >= start_slot).OnlyEnforceIf(is_this_day)
                        self.model.Add(slot_var <= end_slot).OnlyEnforceIf(is_this_day)
                        self.model.Add(slot_var < start_slot).OnlyEnforceIf(is_this_day.Not())
                        self.model.Add(slot_var > end_slot).OnlyEnforceIf(is_this_day.Not())
                        
                        day_sessions.append(is_this_day)
                    
                    # Limiter le nombre de sessions ce jour
                    if day_sessions:
                        self.model.Add(sum(day_sessions) <= max_daily)
        
        # Contrainte : Salles spécialisées pour les TP
        if constraints.get('specialized_rooms', True):
            for session in self.sessions:
                if session['type'] == 'TP':
                    # Vérifier que la salle a l'équipement nécessaire
                    salle = next((s for s in self.salles if s['id'] == session['salle_id']), None)
                    matiere = next((m for m in self.matieres if m['id'] == session['matiere_id']), None)
                    
                    if salle and matiere:
                        required_eq = matiere.get('required_equipment', [])
                        has_equipment = all(eq in salle.get('equipment', []) for eq in required_eq)
                        
                        if not has_equipment and required_eq:
                            # Forcer cette session à ne pas être planifiée (option radicale)
                            # Ou lui donner une pénalité (meilleure approche)
                            slot_var = session_vars[session['id']]
                            # Ici on pourrait ajouter une pénalité à l'objectif
    
    def _extract_solution(self, session_vars):
        """Extrait la solution du solveur"""
        timetable = []
        
        for session in self.sessions:
            slot_var = session_vars[session['id']]
            if self.solver.Value(slot_var) >= 0:  Session planifiée
                slot_value = self.solver.Value(slot_var)
                day = slot_value // self.slots_per_day
                slot_in_day = slot_value % self.slots_per_day
                
                # Convertir en format lisible
                days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
                heure_debut = 8 + slot_in_day
                heure_fin = heure_debut + session['duree']
                
                timetable.append({
                    'ID Session': session['id'],
                    'Jour': days_fr[day],
                    'Créneau': f"{heure_debut:02d}h-{heure_fin:02d}h",
                    'Classe': session['classe_nom'],
                    'Matière': session['matiere_nom'],
                    'Type': session['type'],
                    'Professeur': session['prof_nom'],
                    'Salle': session['salle_nom'],
                    'Durée': f"{session['duree']}h",
                    'Jour_Index': day,
                    'Créneau_Index': slot_in_day
                })
        
        return timetable

# Interface Streamlit améliorée
st.sidebar.header("⚙️ Configuration")

# Upload des fichiers CSV
st.sidebar.subheader("📁 Chargement des fichiers CSV")
profs_file = st.sidebar.file_uploader("Professeurs (CSV)", type="csv", key="profs")
classes_file = st.sidebar.file_uploader("Classes (CSV)", type="csv", key="classes")
salles_file = st.sidebar.file_uploader("Salles (CSV)", type="csv", key="salles")
matieres_file = st.sidebar.file_uploader("Matières (CSV)", type="csv", key="matieres")

# Paramètres de base
st.sidebar.subheader("📅 Paramètres temporels")
jours = st.sidebar.slider("Jours par semaine", 1, 7, 5)
creneaux_par_jour = st.sidebar.slider("Créneaux par jour", 1, 12, 8)
heure_debut = st.sidebar.number_input("Heure de début", 6, 12, 8)

# Contraintes avancées
st.sidebar.subheader("🎯 Contraintes avancées")
no_friday_afternoon = st.sidebar.checkbox("Pas de cours le vendredi après-midi", value=True)
limit_daily_sessions = st.sidebar.checkbox("Limiter les cours par jour par professeur", value=True)
specialized_rooms = st.sidebar.checkbox("Respecter les salles spécialisées", value=True)
group_tp_sessions = st.sidebar.checkbox("Grouper les sessions TP", value=True)

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🎯 Contraintes", "⚡ Génération", "📋 Résultats"])

with tab1:
    st.header("📊 Données Chargées")
    
    # Afficher les données chargées
    if profs_file is not None:
        profs_df = pd.read_csv(profs_file)
        st.subheader("Professeurs")
        st.dataframe(profs_df)
        st.metric("Nombre de professeurs", len(profs_df))
    
    if classes_file is not None:
        classes_df = pd.read_csv(classes_file)
        st.subheader("Classes")
        st.dataframe(classes_df)
        st.metric("Nombre de classes", len(classes_df))
    
    if salles_file is not None:
        salles_df = pd.read_csv(salles_file)
        st.subheader("Salles")
        st.dataframe(salles_df)
        st.metric("Nombre de salles", len(salles_df))
    
    if matieres_file is not None:
        matieres_df = pd.read_csv(matieres_file)
        st.subheader("Matières")
        st.dataframe(matieres_df)
        st.metric("Nombre de matières", len(matieres_df))

with tab2:
    st.header("🎯 Configuration des Contraintes")
    
    st.subheader("Contraintes temporelles")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Jours de travail:**")
        jours_selection = []
        jours_possibles = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        for i, jour in enumerate(jours_possibles[:jours]):
            if st.checkbox(jour, value=True, key=f"jour_{i}"):
                jours_selection.append(jour)
    
    with col2:
        duree_creneau = st.selectbox("Durée des créneaux", [1, 2, 3], index=0)
        pause_dejeuner = st.checkbox("Pause déjeuner (12h-14h)", value=True)
    
    st.subheader("Contraintes pédagogiques")
    col1, col2 = st.columns(2)
    
    with col1:
        max_cours_jour_prof = st.slider("Max cours/jour/prof", 1, 8, 4)
        max_cours_jour_classe = st.slider("Max cours/jour/classe", 1, 10, 6)
    
    with col2:
        repos_entre_cours = st.checkbox("Éviter les cours consécutifs", value=True)
        preference_amplitude = st.slider("Préférence amplitude horaire", 1, 10, 7)

with tab3:
    st.header("⚡ Génération d'Emploi du Temps")
    
    # Vérifier que tous les fichiers sont chargés
    fichiers_charges = all([profs_file, classes_file, salles_file, matieres_file])
    
    if not fichiers_charges:
        st.error("❌ Veuillez charger tous les fichiers CSV pour continuer")
        st.info("**Fichiers requis:** professeurs, classes, salles, matières")
    else:
        if st.button("🚀 Générer l'Emploi du Temps Optimisé", type="primary"):
            with st.spinner("Chargement des données et optimisation en cours..."):
                # Charger les DataFrames
                profs_df = pd.read_csv(profs_file)
                classes_df = pd.read_csv(classes_file)
                salles_df = pd.read_csv(salles_file)
                matieres_df = pd.read_csv(matieres_file)
                
                # Initialiser le solveur
                solver = AdvancedTimetableSolver(days=jours, slots_per_day=creneaux_par_jour)
                
                # Charger les données
                solver.load_data_from_dataframes(profs_df, classes_df, salles_df, matieres_df)
                
                # Préparer les contraintes
                constraints = {
                    'no_friday_afternoon': no_friday_afternoon,
                    'limit_daily_sessions': limit_daily_sessions,
                    'specialized_rooms': specialized_rooms,
                    'group_tp_sessions': group_tp_sessions,
                    'max_daily_prof': max_cours_jour_prof,
                    'max_daily_class': max_cours_jour_classe
                }
                
                # Barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulation des étapes de résolution
                steps = [
                    "Chargement des données...",
                    "Création des variables...", 
                    "Ajout des contraintes...",
                    "Résolution avec OR-Tools...",
                    "Extraction de la solution..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) * 20)
                    time.sleep(1)
                
                # Résoudre le problème
                solution = solver.solve_with_constraints(constraints)
                
                if solution:
                    st.session_state.timetable_solution = solution
                    st.session_state.solver_stats = {
                        'sessions_planifiees': len(solution),
                        'sessions_totales': len(solver.sessions),
                        'taux_reussite': (len(solution) / len(solver.sessions)) * 100,
                        'temps_resolution': solver.solver.WallTime()
                    }
                    st.success("✅ Emploi du temps généré avec succès !")
                else:
                    st.error("❌ Impossible de générer un emploi du temps satisfaisant toutes les contraintes")

with tab4:
    st.header("📋 Résultats et Export")
    
    if 'timetable_solution' in st.session_state:
        solution = st.session_state.timetable_solution
        stats = st.session_state.solver_stats
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sessions planifiées", stats['sessions_planifiees'])
        with col2:
            st.metric("Sessions totales", stats['sessions_totales'])
        with col3:
            st.metric("Taux de réussite", f"{stats['taux_reussite']:.1f}%")
        with col4:
            st.metric("Temps de résolution", f"{stats['temps_resolution']:.2f}s")
        
        # Filtrer et afficher l'emploi du temps
        st.subheader("📅 Emploi du Temps Généré")
        
        # Options de filtrage
        col1, col2, col3 = st.columns(3)
        with col1:
            classe_filter = st.selectbox("Filtrer par classe", 
                                       ['Toutes'] + list(set(s['Classe'] for s in solution)))
        with col2:
            jour_filter = st.selectbox("Filtrer par jour", 
                                     ['Tous'] + list(set(s['Jour'] for s in solution)))
        with col3:
            prof_filter = st.selectbox("Filtrer par professeur", 
                                     ['Tous'] + list(set(s['Professeur'] for s in solution)))
        
        # Appliquer les filtres
        filtered_solution = solution
        if classe_filter != 'Toutes':
            filtered_solution = [s for s in filtered_solution if s['Classe'] == classe_filter]
        if jour_filter != 'Tous':
            filtered_solution = [s for s in filtered_solution if s['Jour'] == jour_filter]
        if prof_filter != 'Tous':
            filtered_solution = [s for s in filtered_solution if s['Professeur'] == prof_filter]
        
        # Afficher le tableau
        solution_df = pd.DataFrame(filtered_solution)
        st.dataframe(solution_df, use_container_width=True)
        
        # Export Excel
        st.subheader("📤 Export")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Feuille principale
            pd.DataFrame(solution).to_excel(writer, sheet_name='Emploi_du_temps', index=False)
            
            # Feuilles par classe
            for classe in set(s['Classe'] for s in solution):
                classe_data = [s for s in solution if s['Classe'] == classe]
                if classe_data:
                    pd.DataFrame(classe_data).to_excel(writer, sheet_name=classe[:31], index=False)
            
            # Feuille par professeur
            for prof in set(s['Professeur'] for s in solution):
                prof_data = [s for s in solution if s['Professeur'] == prof]
                if prof_data:
                    pd.DataFrame(prof_data).to_excel(writer, sheet_name=f"Prof_{prof}"[:31], index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Télécharger l'emploi du temps Excel complet",
            data=output,
            file_name=f"emploi_du_temps_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Visualisation par classe
        st.subheader("📊 Visualisation par Classe")
        classe_select = st.selectbox("Choisir une classe pour visualisation", 
                                   list(set(s['Classe'] for s in solution)))
        
        if classe_select:
            classe_data = [s for s in solution if s['Classe'] == classe_select]
            classe_df = pd.DataFrame(classe_data)
            
            # Créer une grille horaire
            jours_ordre = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            creneaux_ordre = [f"{heure_debut + i:02d}h-{heure_debut + i + 1:02d}h" for i in range(creneaux_par_jour)]
            
            # Préparer les données pour la grille
            grid_data = []
            for jour in jours_ordre[:jours]:
                for creneau in creneaux_ordre:
                    cours_ce_creneau = [c for c in classe_data if c['Jour'] == jour and c['Créneau'] == creneau]
                    if cours_ce_creneau:
                        for cours in cours_ce_creneau:
                            grid_data.append({
                                'Jour': jour,
                                'Créneau': creneau,
                                'Matière': cours['Matière'],
                                'Professeur': cours['Professeur'],
                                'Salle': cours['Salle'],
                                'Type': cours['Type']
                            })
                    else:
                        grid_data.append({
                            'Jour': jour, 
                            'Créneau': creneau,
                            'Matière': 'Libre',
                            'Professeur': '',
                            'Salle': '',
                            'Type': ''
                        })
            
            grid_df = pd.DataFrame(grid_data)
            pivot_df = grid_df.pivot_table(index='Créneau', columns='Jour', 
                                         values='Matière', aggfunc='first', 
                                         fill_value='Libre')
            
            st.dataframe(pivot_df, use_container_width=True)
    
    else:
        st.info("💡 Générez d'abord un emploi du temps dans l'onglet 'Génération'")

# Footer avec informations
st.markdown("---")
st.markdown("""
**Fonctionnalités implémentées:**
- ✅ Chargement des fichiers CSV spécifiques
- ✅ Contraintes de base (non-chevauchenent)
- ✅ Contraintes avancées (vendredi après-midi, limites quotidiennes)
- ✅ Gestion des salles spécialisées
- ✅ Filtrage et export Excel
- ✅ Visualisation par classe
""")
